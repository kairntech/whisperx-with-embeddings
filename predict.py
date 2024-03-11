import os
import time
from itertools import groupby

from whisperx import load_audio
from whisperx.audio import SAMPLE_RATE
# Better define XDG_CACHE_HOME as env variable in the docker container
# os.environ['HF_HOME'] = '/src/hf_models'
# os.environ['TORCH_HOME'] = '/src/torch_models'
from cog import BasePredictor, Input, Path, BaseModel
from pydub import AudioSegment
from typing import Any, Optional, Union
import torch
import whisperx
import faster_whisper
import numpy as np
import pandas as pd
from pyannote.audio import Pipeline

compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
device = "cuda"

class Output(BaseModel):
    segments: Any
    embeddings: Any
    detected_language: str

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"
        asr_options = {
            "temperatures": [int(os.getenv("TEMPERATURE", "0"))],
            "initial_prompt": os.getenv("INITIAL_PROMPT", None)
        }

        vad_options = {
            "vad_onset": float(os.getenv("VAD_ONSET", "0.500")),
            "vad_offset": float(os.getenv("VAD_OFFSET", "0.363"))
        }
        self.model = whisperx.load_model(os.getenv("WHISPER_MODEL", "tiny"), self.device,
                                         language=os.getenv("LANG", "fr"), compute_type=compute_type,
                                         asr_options=asr_options, vad_options=vad_options)
        self.alignment_model, self.metadata = whisperx.load_align_model(language_code=os.getenv("LANG", "fr"),
                                                                        device=self.device)
        self.diarize_model = DiarizationWithEmbeddingsPipeline(model_name='pyannote/speaker-diarization-3.1',
                                                          use_auth_token=os.getenv("HF_TOKEN"), device=device)
    def predict(
            self,
            audio_file: Path = Input(description="Audio file"),
            initial_prompt: str = Input(
                description="Optional text to provide as a prompt for the first window",
                default=None),
            temperature: float = Input(
                description="Temperature to use for sampling",
                default=0),
            batch_size: int = Input(
                description="Parallelization of input audio transcription",
                default=64),
            align_output: bool = Input(
                description="Aligns whisper output to get accurate word-level timestamps",
                default=True),
            diarization: bool = Input(
                description="Assign speaker ID labels",
                default=True),
            group_adjacent_speakers: bool = Input(
                description="Group adjacent segments with same speakers",
                default=True),
            min_speakers: int = Input(
                description="Minimum number of speakers if diarization is activated (leave blank if unknown)",
                default=None),
            max_speakers: int = Input(
                description="Maximum number of speakers if diarization is activated (leave blank if unknown)",
                default=None),
            return_embeddings: bool = Input(
                description="Return representative speaker embeddings",
                default=True),
            debug: bool = Input(
                    description="Print out compute/inference times and memory usage information",
                    default=False)
    ) -> Output:
        """Run a single prediction on the model"""
        with torch.inference_mode():
            new_asr_options = self.model.options._asdict()
            if (initial_prompt and new_asr_options["initial_prompt"] != initial_prompt) or temperature not in \
                    new_asr_options[
                        "temperatures"]:
                new_asr_options["initial_prompt"] = initial_prompt
                new_asr_options["temperatures"] = [temperature]
                new_options = faster_whisper.transcribe.TranscriptionOptions(**new_asr_options)
                self.model.options = new_options
            audio = whisperx.load_audio(audio_file)
            result = self.model.transcribe(audio, batch_size=batch_size)
            detected_language = result["language"]
            if align_output:
                if detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_TORCH or detected_language in whisperx.alignment.DEFAULT_ALIGN_MODELS_HF:
                    result = self.align(audio, result, debug)
                else:
                    print(f"Cannot align output as language {detected_language} is not supported for alignment")

            if diarization:
                result = self.diarize(audio, result, debug, min_speakers, max_speakers, return_embeddings)
                if group_adjacent_speakers:
                    result = self.group_speakers(result, debug)
            if debug:
                print(f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")

        segments = result["segments"]
        for segment in segments:
            segment.pop("words", None)
        return Output(
            segments=result["segments"],
            embeddings=result.get("embeddings", None),
            detected_language=detected_language
        )

    def group_speakers(self, result, debug):
        start_time = time.time_ns() / 1e6
        segments = result["segments"]
        grouped_segments = []
        for key, group in groupby(segments, lambda seg: seg.get('speaker', 'Unknown')):
            segs = list(group)
            grouped_segment = segs[0]
            grouped_segment['end'] = segs[-1]['end']
            grouped_segment['text'] = "\n".join(seg['text'] for seg in segs)
            grouped_segment.pop('words', None)
            grouped_segments.append(grouped_segment)

        if debug:
            elapsed_time = time.time_ns() / 1e6 - start_time
            print(f"Duration to group speakers: {elapsed_time:.2f} ms")

        result['segments'] = grouped_segments
        return result

    def align(self, audio, result, debug):
        start_time = time.time_ns() / 1e6
        result = whisperx.align(result["segments"], self.alignment_model, self.metadata, audio, device,
                                return_char_alignments=False)

        if debug:
            elapsed_time = time.time_ns() / 1e6 - start_time
            print(f"Duration to align output: {elapsed_time:.2f} ms")

        return result

    def diarize(self, audio, result, debug, min_speakers, max_speakers, return_embeddings):
        start_time = time.time_ns() / 1e6

        diarize_segments, embeddings = self.diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers, return_embeddings=return_embeddings)

        result = whisperx.assign_word_speakers(diarize_segments, result)
        if return_embeddings:
            result['embeddings'] = embeddings
            print(embeddings)

        if debug:
            elapsed_time = time.time_ns() / 1e6 - start_time
            print(f"Duration to diarize segments: {elapsed_time:.2f} ms")

        return result


def get_audio_duration(file_path):
    return len(AudioSegment.from_file(file_path))

class DiarizationWithEmbeddingsPipeline:
    def __init__(
            self,
            model_name="pyannote/speaker-diarization-3.1",
            use_auth_token=None,
            device: Optional[Union[str, torch.device]] = "cpu",
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.model = Pipeline.from_pretrained(model_name, use_auth_token=use_auth_token).to(device)

    def __call__(self, audio: Union[str, np.ndarray], min_speakers=None, max_speakers=None, return_embeddings=False):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }

        if not return_embeddings:
            segments = self.model(audio_data, min_speakers=min_speakers, max_speakers=max_speakers, return_embeddings=False)
            embeddings = None
        else:
            segments, embeddings = self.model(audio_data, min_speakers=min_speakers, max_speakers=max_speakers, return_embeddings=True)
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
        return diarize_df, embeddings




