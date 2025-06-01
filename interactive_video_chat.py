#!/usr/bin/env python3
"""
Interactive chat with Llama about video analysis results
"""

import json
import sys
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class VideoAnalysisChat:
    def __init__(self, analysis_file):
        """Initialize chat with video analysis context"""
        self.client = OpenAI(
            api_key=os.getenv('LLAMA4_API_KEY'),
            base_url="https://api.llama.com/compat/v1/"
        )
        
        # Load analysis results
        with open(analysis_file, 'r') as f:
            self.analysis_data = json.load(f)
        
        # Build context
        self.context = self._build_context()
        self.conversation_history = []
        
    def _build_context(self):
        """Build context from analysis data"""
        data = self.analysis_data
        
        context = f"""
VIDEO ANALYSIS CONTEXT:
======================
Video: {data['video_path']}
Frames: {data['frames_extracted']}
Transcript ({data['transcript_length']} chars): {data['transcript']}

ANALYSIS RESULTS:
"""
        
        # Add all analysis results
        analysis = data.get('analysis', {})
        for key, value in analysis.items():
            context += f"\n{key.upper()}:\n{value}\n"
            
        return context
    
    def chat_loop(self):
        """Main interactive chat loop"""
        print("🎬 Video Analysis Chat - Ask me anything about the video!")
        print("Commands: 'quit', 'exit', 'clear', 'context', 'help'")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                    
                elif user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("🧹 Conversation history cleared!")
                    continue
                    
                elif user_input.lower() == 'context':
                    print("📋 Video Context:")
                    print(f"Video: {self.analysis_data['video_path']}")
                    print(f"Frames: {self.analysis_data['frames_extracted']}")
                    print(f"Transcript: {self.analysis_data['transcript_length']} chars")
                    continue
                    
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                    
                elif not user_input:
                    continue
                
                # Get response from Llama
                response = self._get_llama_response(user_input)
                print(f"\n🤖 Llama: {response}")
                
                # Add to history
                self.conversation_history.append({
                    "user": user_input,
                    "assistant": response
                })
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _get_llama_response(self, user_question):
        """Get response from Llama with context"""
        
        # Build messages with context and history
        messages = [
            {
                "role": "system",
                "content": f"""You are an AI assistant analyzing a video. Use this context to answer questions:

{self.context}

Answer questions about the video content, analysis, people, conversation, or any insights. Be helpful and specific."""
            }
        ]
        
        # Add conversation history
        for exchange in self.conversation_history[-5:]:  # Last 5 exchanges
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["assistant"]})
        
        # Add current question
        messages.append({"role": "user", "content": user_question})
        
        try:
            response = self.client.chat.completions.create(
                model="Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error getting response: {e}"
    
    def _show_help(self):
        """Show help commands"""
        print("""
📚 Available Commands:
- quit/exit/q: End the chat
- clear: Clear conversation history  
- context: Show video details
- help: Show this help

💡 Example Questions:
- "What were the main topics discussed?"
- "How did the participants' body language change?"
- "What networking advice would you give?"
- "Summarize the key insights"
- "What questions should have been asked?"
- "Rate the meeting effectiveness"
""")

def main():
    if len(sys.argv) < 2:
        print("Usage: python interactive_video_chat.py <analysis_json_file>")
        print("Example: python interactive_video_chat.py videoNetworking_llama_analysis.json")
        return
    
    analysis_file = sys.argv[1]
    
    if not os.path.exists(analysis_file):
        print(f"❌ Analysis file not found: {analysis_file}")
        print("Run video analysis first:")
        print("python llama_video_analyzer.py data/video.MOV --mode transcript_only")
        return
    
    chat = VideoAnalysisChat(analysis_file)
    chat.chat_loop()

if __name__ == "__main__":
    main() 