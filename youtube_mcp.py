import json
import re
from typing import Dict, List
from youtube_transcript_api import YouTubeTranscriptApi
from mcp.server.fastmcp import FastMCP


class YouTubeTranscriptService:
    """Service class to handle YouTube transcript extraction."""

    def __init__(self):
        self.api = YouTubeTranscriptApi()

    @staticmethod
    def extract_video_id(url: str) -> str:
        """Extract video ID from a YouTube URL."""
        pattern = r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)'
        match = re.search(pattern, url)
        return match.group(1) if match else None

    def fetch_transcript(self, url: str, preferred_languages: List[str] = ["en"]) -> Dict:
        """Fetch transcript for a given YouTube video URL."""
        video_id = self.extract_video_id(url)
        if not video_id:
            return {"success": False, "error": "Invalid YouTube URL"}

        try:
            transcript_list = self.api.list(video_id)

            # Try preferred languages
            transcript = None
            for lang in preferred_languages:
                for t in transcript_list:
                    if t.language_code == lang:
                        transcript = t
                        break
                if transcript:
                    break

            # Fallback: first available transcript
            if not transcript:
                transcripts = list(transcript_list)
                if transcripts:
                    transcript = transcripts[0]
                else:
                    return {"success": False, "error": "No transcripts available"}

            transcript_data = transcript.fetch()
            full_text = " ".join([entry.text for entry in transcript_data])

            return {
                "success": True,
                "transcript": full_text,
                "language": transcript.language_code,
                "transcript_type": "generated" if transcript.is_generated else "manual",
                "video_id": video_id,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}


# Initialize MCP server
mcp = FastMCP("youtube_transcript")
yt_service = YouTubeTranscriptService()


@mcp.tool(name="get_transcript", description="Get transcript by URL")
def get_transcript(url: str, preferred_languages: List[str] = ["en"]) -> str:
    """
    MCP tool: Get transcript of a YouTube video.
    """
    result = yt_service.fetch_transcript(url, preferred_languages)
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    '''import sys
    if len(sys.argv) > 1 and sys.argv[1] in ["test", "--test"]:
        print("▶ Running in TEST mode...")
        test_url = "https://youtu.be/I2wURDqiXdM?si=ytma61TmMeP2YPZX"
        print(f"Fetching transcript for: {test_url}")
        result = yt_service.fetch_transcript(test_url, preferred_languages=["en", "hi"])
        print("Raw result:", json.dumps(result, indent=2)[:500], "...")
        if result["success"]:
            print(f"Video ID: {result['video_id']}")
            print(f"Language: {result['language']}")
            print(f"Transcript type: {result['transcript_type']}")
            print("Transcript sample:")
            print(result["transcript"][:500] + "..." if len(result["transcript"]) > 500 else result["transcript"])
        else:
            print("❌ Error:", result["error"])
    else:
        print("▶ Starting YouTube MCP server (stdio transport)...")
    '''
    mcp.run(transport="stdio")

