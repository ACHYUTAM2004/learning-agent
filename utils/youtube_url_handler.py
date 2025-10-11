from youtube_transcript_api import YouTubeTranscriptApi
import re

def get_video_id(youtube_url):
    """Extracts the video ID from a YouTube URL."""
    # This regex handles standard, shortened, and embed URLs
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, youtube_url)
    return match.group(1) if match else None

def get_transcript(youtube_url):
    """
    Fetches the transcript for a given YouTube video URL and returns it as a single string.
    """
    video_id = get_video_id(youtube_url)
    if not video_id:
        return None, "Invalid YouTube URL provided."

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        # Combine the transcript parts into a single block of text
        full_transcript = " ".join([item['text'] for item in transcript_list])
        return full_transcript, None
    except Exception as e:
        return None, f"Could not retrieve transcript: {e}"