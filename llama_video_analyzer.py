#!/usr/bin/env python3
"""
Complete video analysis with Llama: Extract frames + transcribe + analyze
"""

import sys
import os
import json
from datetime import datetime
from typing import List, Dict, Any
from video_frame_extractor_element import VideoFrameExtractorElement
from transcribe_video import transcribe_video_simple
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LlamaVideoAnalyzer:
    def __init__(self, api_key=None):
        """Initialize with Llama API client"""
        self.client = OpenAI(
            base_url="https://api.llama.com/compat/v1/",
            api_key=api_key or os.getenv('LLAMA4_API_KEY')
        )
        
        # Initialize frame extractor
        self.frame_extractor = VideoFrameExtractorElement()
        
        if not os.getenv('LLAMA4_API_KEY'):
            print("⚠️  Warning: LLAMA4_API_KEY not set. Set with: export LLAMA4_API_KEY='your-key'")
    
    def extract_frames(self, video_path, interval=20):
        """Extract frames using existing extractor"""
        print(f"📸 Extracting frames every {interval} seconds...")
        frames = self.frame_extractor.process(video_path, interval)
        print(f"✅ Extracted {len(frames)} frames")
        return frames
    
    def transcribe_video(self, video_path, model="base"):
        """Transcribe video using existing transcriber"""
        print(f"🎵 Transcribing with Whisper {model} model...")
        transcript = transcribe_video_simple(video_path, model)
        if transcript:
            print(f"✅ Transcribed {len(transcript)} characters")
        return transcript or ""
    
    def analyze_frame_with_context(self, frame, transcript, custom_prompt=None):
        """Analyze single frame with transcript context"""
        
        prompt = custom_prompt or f"""
        Analyze this video frame at {frame['timestamp']} seconds.
        
        TRANSCRIPT CONTEXT:
        {transcript}
        
        Please analyze:
        1. What's happening visually at this moment
        2. How the visual relates to the spoken content
        3. Body language and non-verbal communication
        4. Key insights from this specific moment
        5. Overall meeting dynamics visible in this frame
        """
        
        try:
            response = self.client.chat.completions.create(
                model="Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frame['image_base64']}"
                            }
                        }
                    ]
                }]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error analyzing frame: {e}"
    
    def analyze_all_frames_together(self, frames, transcript):
        """Analyze all frames together in one request"""
        
        content = [{
            "type": "text",
            "text": f"""
            Analyze this complete video using both visual frames and audio transcript.
            
            TRANSCRIPT:
            {transcript}
            
            VISUAL FRAMES (chronological order):
            """
        }]
        
        # Add all frames
        for i, frame in enumerate(frames):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame['image_base64']}"
                }
            })
            content.append({
                "type": "text",
                "text": f"Frame {i+1} at {frame['timestamp']}s"
            })
        
        content.append({
            "type": "text",
            "text": """
            Provide comprehensive analysis:
            1. Overall conversation flow and dynamics
            2. How visual and audio elements work together
            3. Key moments where visuals enhance understanding
            4. Non-verbal communication patterns
            5. Meeting effectiveness and networking insights
            6. Recommendations based on the interaction
            """
        })
        
        try:
            response = self.client.chat.completions.create(
                model="Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=[{"role": "user", "content": content}]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error in comprehensive analysis: {e}"
    
    def analyze_transcript_only(self, transcript):
        """Analyze just the transcript for text-based insights"""
        
        prompt = f"""
        Analyze this video conversation transcript:
        
        {transcript}
        
        Provide insights about:
        1. Main topics and themes discussed
        2. Key decisions or action items mentioned
        3. Conversation flow and participants' roles
        4. Networking effectiveness and outcomes
        5. Professional communication quality
        6. Recommendations for improvement
        """
        
        try:
            response = self.client.chat.completions.create(
                model="Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error analyzing transcript: {e}"
    
    def analyze_video_complete(self, video_path, frame_interval=20, whisper_model="base", analysis_mode="comprehensive"):
        """Complete video analysis pipeline"""
        
        print(f"🎬 Analyzing video: {video_path}")
        print("=" * 60)
        
        results = {
            "video_path": video_path,
            "settings": {
                "frame_interval": frame_interval,
                "whisper_model": whisper_model,
                "analysis_mode": analysis_mode
            }
        }
        
        # Step 1: Extract frames
        frames = self.extract_frames(video_path, frame_interval)
        results["frames_extracted"] = len(frames)
        
        # Step 2: Transcribe audio  
        transcript = self.transcribe_video(video_path, whisper_model)
        results["transcript"] = transcript
        results["transcript_length"] = len(transcript)
        
        if not transcript:
            print("❌ No transcript available, analyzing frames only")
            analysis_mode = "frames_only"
        
        # Step 3: Analyze with Llama
        print("🤖 Analyzing with Llama...")
        
        if analysis_mode == "comprehensive":
            # Individual frame analysis
            frame_analyses = []
            for i, frame in enumerate(frames):
                print(f"  📸 Analyzing frame {i+1}/{len(frames)} at {frame['timestamp']}s...")
                analysis = self.analyze_frame_with_context(frame, transcript)
                frame_analyses.append({
                    "timestamp": frame['timestamp'],
                    "analysis": analysis
                })
            
            # Overall analysis
            print("  🔍 Running comprehensive analysis...")
            overall_analysis = self.analyze_all_frames_together(frames, transcript)
            
            # Transcript-only analysis
            print("  📝 Analyzing transcript...")
            transcript_analysis = self.analyze_transcript_only(transcript)
            
            results["analysis"] = {
                "individual_frames": frame_analyses,
                "comprehensive": overall_analysis,
                "transcript_only": transcript_analysis
            }
            
        elif analysis_mode == "frames_only":
            frame_analyses = []
            for frame in frames:
                analysis = self.analyze_frame_with_context(frame, transcript or "No transcript available")
                frame_analyses.append({
                    "timestamp": frame['timestamp'],
                    "analysis": analysis
                })
            results["analysis"] = {"frames_only": frame_analyses}
            
        elif analysis_mode == "transcript_only":
            results["analysis"] = {"transcript_only": self.analyze_transcript_only(transcript)}
            
        elif analysis_mode == "overview":
            results["analysis"] = {"overview": self.analyze_all_frames_together(frames, transcript)}
        
        print("✅ Analysis complete!")
        return results
    
    def save_results(self, results, output_file=None):
        """Save analysis results to file"""
        
        if not output_file:
            base_name = os.path.splitext(os.path.basename(results["video_path"]))[0]
            output_file = f"{base_name}_llama_analysis.json"
        
        # Create readable text version
        text_file = output_file.replace('.json', '.txt')
        
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("LLAMA VIDEO ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Video: {results['video_path']}\n")
            f.write(f"Frames extracted: {results['frames_extracted']}\n")
            f.write(f"Transcript length: {results['transcript_length']} characters\n")
            f.write(f"Settings: {results['settings']}\n\n")
            
            # Write analysis results
            analysis = results.get('analysis', {})
            
            if 'transcript_only' in analysis:
                f.write("TRANSCRIPT ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                f.write(analysis['transcript_only'] + "\n\n")
            
            if 'individual_frames' in analysis:
                f.write("INDIVIDUAL FRAME ANALYSIS:\n")
                f.write("-" * 35 + "\n")
                for frame_result in analysis['individual_frames']:
                    f.write(f"\nFrame at {frame_result['timestamp']}s:\n")
                    f.write(frame_result['analysis'] + "\n")
                    f.write("-" * 20 + "\n")
            
            if 'comprehensive' in analysis:
                f.write("\nCOMPREHENSIVE ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                f.write(analysis['comprehensive'] + "\n\n")
            
            if 'overview' in analysis:
                f.write("OVERVIEW ANALYSIS:\n")
                f.write("-" * 25 + "\n")
                f.write(analysis['overview'] + "\n\n")
        
        # Save JSON version for programmatic access
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to:")
        print(f"  📄 {text_file} (human readable)")
        print(f"  📋 {output_file} (JSON data)")

def main():
    if len(sys.argv) < 2:
        print("Usage: python llama_video_analyzer.py <video_file> [options]")
        print("\nOptions:")
        print("  --interval SECONDS     Frame extraction interval (default: 20)")
        print("  --whisper MODEL        Whisper model: tiny,base,small,medium,large (default: base)")
        print("  --mode MODE            Analysis mode: comprehensive,frames_only,transcript_only,overview (default: comprehensive)")
        print("  --output FILE          Output file prefix")
        print("\nExamples:")
        print("  python llama_video_analyzer.py data/test.MOV")
        print("  python llama_video_analyzer.py data/test.MOV --interval 10 --whisper medium")
        print("  python llama_video_analyzer.py data/test.MOV --mode overview")
        return
    
    # Parse arguments
    video_path = sys.argv[1]
    frame_interval = 20
    whisper_model = "base"
    analysis_mode = "comprehensive"
    output_file = None
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--interval" and i+1 < len(sys.argv):
            frame_interval = int(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == "--whisper" and i+1 < len(sys.argv):
            whisper_model = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "--mode" and i+1 < len(sys.argv):
            analysis_mode = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "--output" and i+1 < len(sys.argv):
            output_file = sys.argv[i+1]
            i += 2
        else:
            i += 1
    
    # Validate inputs
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return
    
    if analysis_mode not in ["comprehensive", "frames_only", "transcript_only", "overview"]:
        print(f"❌ Error: Invalid analysis mode: {analysis_mode}")
        return
    
    # Run analysis
    analyzer = LlamaVideoAnalyzer()
    results = analyzer.analyze_video_complete(
        video_path, frame_interval, whisper_model, analysis_mode
    )
    
    # Save results
    analyzer.save_results(results, output_file)
    
    print("\n🎉 Video analysis complete!")

if __name__ == "__main__":
    main() 