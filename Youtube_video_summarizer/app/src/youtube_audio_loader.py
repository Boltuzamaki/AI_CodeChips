from langchain_community.document_loaders.blob_loaders.youtube_audio import (
    YoutubeAudioLoader,
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers.audio import OpenAIWhisperParserLocal, OpenAIWhisperParser
import os 

def youtube_transcriber(youtube_video_link, local=True):
    urls = [youtube_video_link]

    save_dir = os.path.expanduser("~/Downloads/YouTube")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if local:
        loader = GenericLoader(
            YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParserLocal()
        )
    else:
        loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser())
    
    docs = loader.load()
    
    for file_name in os.listdir(save_dir):
        file_path = os.path.join(save_dir, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    if not os.listdir(save_dir):
        os.rmdir(save_dir)

    return docs
