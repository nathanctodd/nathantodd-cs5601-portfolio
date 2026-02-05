from youtube_transcript_api import YouTubeTranscriptApi
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent
import dotenv
from langchain.agents import create_agent
dotenv.load_dotenv()  # Load environment variables from .env file

@tool
def get_youtube_transcript(video_id: str) -> str:
    """Fetch the transcript of a YouTube video by video ID."""
    transcript = YouTubeTranscriptApi.list(video_id)
    return " ".join([entry['text'] for entry in transcript])

# Create agent with tool
llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, [get_youtube_transcript])

# Test it
result = agent.invoke({
    "messages": [("user", "Get the transcript for video dQw4w9WgXcQ and summarize it")]
})
