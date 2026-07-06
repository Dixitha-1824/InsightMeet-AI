from utils.audio_processor import process_input
from core.transcriber import transcribe_all


src="https://www.youtube.com/watch?v=Bm6t7bw8Iqc"
chunks=process_input(src)
print(transcribe_all(chunks))