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
    Fetches the transcript and returns it along with the clean video_id.
    """
    video_id = get_video_id(youtube_url)
    if not video_id:
        # Return three values to maintain a consistent function signature
        return None, None, "Invalid YouTube URL provided."

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = " ".join([item['text'] for item in transcript_list])
        # Return the transcript, the video_id, and no error
        return full_transcript, video_id, None
    except Exception as e:
        return None, None, f"Could not retrieve transcript: {e}"