import os

class TextIterator:
    def __init__(self, filepath: str, chunk_size: int = 1024 * 1024):
        """
        This class itterates through a text file and yields a list of words.
        It breaks the itteration into chunks of a fixed size (default 1MB) to avoid loading the whole file into memory.
        Uses a buffer system to make sure we never split words across chunk boundaries.
        """
        self.filepath = filepath
        self.chunk_size = chunk_size

    def __iter__(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            buffer = ""
            
            while True:
                chunk = f.read(self.chunk_size)
                
                if not chunk:
                    if buffer:
                        yield buffer # buffer is always a contiguous chunk with no spaces
                    break

                text_block = buffer + chunk
                last_space_index = text_block.rfind(' ')

                if last_space_index != -1:
                    safe_text = text_block[:last_space_index]
                    buffer = text_block[last_space_index + 1:]
                    yield safe_text.split()
                else:
                    buffer = text_block
