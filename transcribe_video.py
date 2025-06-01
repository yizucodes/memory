import ffmpeg
import torchaudio
import torch
import argparse
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def extract_audio(video_path, audio_output="audio.wav"):
    """Extract mono audio at 16kHz from video using ffmpeg."""
    ffmpeg.input(video_path).output(audio_output, ac=1, ar='16k').run(overwrite_output=True)
    return audio_output

def load_custom_audio(filepath):
    """Load audio using torchaudio and return waveform + sample rate."""
    waveform, sample_rate = torchaudio.load(filepath)
    return waveform.squeeze().numpy(), sample_rate

def process_audio_chunks(audio_array, sampling_rate, processor, model, device):
    """Transcribe long audio by processing it in 30s chunks with 2s overlap."""
    CHUNK_LENGTH_SEC = 30
    OVERLAP_SEC = 2
    chunk_length = CHUNK_LENGTH_SEC * sampling_rate
    overlap_length = OVERLAP_SEC * sampling_rate
    stride_length = chunk_length - overlap_length

    chunks_transcription = []
    position = 0

    while position < len(audio_array):
        chunk_end = min(position + chunk_length, len(audio_array))
        chunk = audio_array[position:chunk_end]

        input_features = processor(
            chunk,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        ).input_features.to(device)

        predicted_ids = model.generate(
            input_features,
            language="en",
            num_beams=5,
            no_repeat_ngram_size=3
        )

        chunk_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        chunks_transcription.append(chunk_text.strip())

        progress = min(100, (position / len(audio_array)) * 100)
        print(f"Chunk progress: {progress:.1f}% - Processed {position/sampling_rate:.1f}s / {len(audio_array)/sampling_rate:.1f}s", end='\r')
        position += stride_length

    print("\nChunk processing complete!")
    return ' '.join(chunks_transcription)

def transcribe_my_audio(file_path, model_name="openai/whisper-base", save_output=True):
    """Main transcription function."""
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Using device: {device}")
    
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)

    audio_array, sampling_rate = load_custom_audio(file_path)
    transcript = process_audio_chunks(audio_array, sampling_rate, processor, model, device)

    if save_output:
        base = os.path.splitext(os.path.basename(file_path))[0]
        out_file = f"{base}_transcript.txt"
        with open(out_file, "w") as f:
            f.write(transcript)
        print(f"\nTranscript saved to: {out_file}")
    else:
        print("\n--- Transcript ---\n")
        print(transcript)

    return transcript

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe a video/audio file using Whisper")
    parser.add_argument("input_file", help="Path to .mp4, .wav, .mp3, etc.")
    parser.add_argument("--model", default="openai/whisper-base", help="Whisper model to use")
    parser.add_argument("--nosave", action="store_true", help="Don’t save output to file")

    args = parser.parse_args()
    input_path = args.input_file

    if input_path.endswith(".mp4") or input_path.endswith(".mov"):
        print(f"Extracting audio from video: {input_path}")
        input_path = extract_audio(input_path)

    transcribe_my_audio(input_path, model_name=args.model, save_output=not args.nosave)
