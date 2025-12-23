#!/usr/bin/env python3
"""
Smart Restructure Data Prep - Auto-matches longforms with clips

Scans a folder of mixed videos, transcribes all, finds which clips
came from which longforms, creates training pairs automatically
"""

import asyncio
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent))

from create_clips_v4_integrated import CreateClipsV4Integrated


class SmartRestructurePrep:
    """Auto-detect and match longforms with clips"""
    
    def __init__(self):
        self.examples_dir = Path("data/restructure_examples")
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        self.v4 = CreateClipsV4Integrated()
    
    async def scan_and_prepare(
        self,
        source_folder: str,
        min_longform_duration: float = 300.0,  # 5 min
        max_clip_duration: float = 120.0  # 2 min
    ):
        """
        Scan folder, transcribe all videos, auto-match clips to longforms
        """
        
        source_path = Path(source_folder)
        
        if not source_path.exists():
            print(f"❌ Folder not found: {source_folder}")
            return
        
        print(f"\n{'='*70}")
        print(f"🔍 SCANNING FOLDER")
        print(f"{'='*70}")
        print(f"\n   Path: {source_path}")
        
        # Find all MP4 files
        video_files = list(source_path.glob("*.mp4")) + list(source_path.glob("*.MP4"))
        
        print(f"\n   Found {len(video_files)} videos")
        
        if not video_files:
            print(f"\n❌ No MP4 files found!")
            return
        
        # Transcribe all videos
        print(f"\n{'='*70}")
        print(f"📝 TRANSCRIBING ALL VIDEOS")
        print(f"{'='*70}")
        
        transcripts = []
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] {video_file.name}")
            
            # Check if transcript already exists
            cache_name = video_file.stem + "_transcript.json"
            cache_path = Path("data/cache/transcripts") / cache_name
            
            if cache_path.exists():
                print(f"   ✅ Loading cached transcript...")
                with open(cache_path) as f:
                    data = json.load(f)
                    segments = data['segments']
            else:
                print(f"   🎙️  Transcribing (may take 3-7 min)...")
                segments = await self.v4._transcribe_with_assemblyai(video_file)
                
                if not segments:
                    print(f"   ❌ Transcription failed, skipping...")
                    continue
                
                # Cache it
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'w') as f:
                    json.dump({
                        'video_path': str(video_file),
                        'segments': segments,
                        'service': 'assemblyai'
                    }, f, indent=2)
                print(f"   ✅ Cached transcript")
            
            # Calculate duration
            duration = segments[-1]['end'] if segments else 0
            
            # Classify as longform or clip
            if duration >= min_longform_duration:
                video_type = 'longform'
            elif duration <= max_clip_duration:
                video_type = 'clip'
            else:
                video_type = 'unknown'
            
            transcripts.append({
                'file': video_file,
                'transcript_path': cache_path,
                'segments': segments,
                'duration': duration,
                'type': video_type,
                'segment_count': len(segments)
            })
            
            print(f"   ✅ {len(segments)} segments, {duration:.1f}s → {video_type.upper()}")
        
        # Separate longforms and clips
        longforms = [t for t in transcripts if t['type'] == 'longform']
        clips = [t for t in transcripts if t['type'] == 'clip']
        
        print(f"\n{'='*70}")
        print(f"📊 CLASSIFICATION")
        print(f"{'='*70}")
        print(f"\n   Longforms: {len(longforms)}")
        for lf in longforms:
            print(f"      • {lf['file'].name} ({lf['duration']/60:.1f} min)")
        
        print(f"\n   Clips: {len(clips)}")
        for clip in clips:
            print(f"      • {clip['file'].name} ({clip['duration']:.1f}s)")
        
        if not longforms:
            print(f"\n❌ No longforms found! (need videos >5min)")
            return
        
        if not clips:
            print(f"\n❌ No clips found! (need videos <2min)")
            return
        
        # Match clips to longforms
        print(f"\n{'='*70}")
        print(f"🔗 MATCHING CLIPS TO LONGFORMS")
        print(f"{'='*70}")
        
        matches = []
        
        for clip in clips:
            best_match = self._find_best_longform(clip, longforms)
            
            if best_match:
                matches.append({
                    'longform': best_match['longform'],
                    'clip': clip,
                    'similarity': best_match['similarity'],
                    'matched_segments': best_match['matched_count']
                })
                
                print(f"\n   ✅ MATCH FOUND:")
                print(f"      Clip: {clip['file'].name}")
                print(f"      Longform: {best_match['longform']['file'].name}")
                print(f"      Similarity: {best_match['similarity']:.0%}")
                print(f"      Matched: {best_match['matched_count']}/{len(clip['segments'])} segments")
            else:
                print(f"\n   ⚠️  NO MATCH:")
                print(f"      Clip: {clip['file'].name}")
                print(f"      (couldn't find source longform)")
        
        if not matches:
            print(f"\n❌ No matches found!")
            print(f"\n💡 This could mean:")
            print(f"   - Clips are from different videos not in folder")
            print(f"   - Transcription quality too low")
            print(f"   - Need to adjust matching threshold")
            return
        
        # Create examples
        print(f"\n{'='*70}")
        print(f"📦 CREATING EXAMPLES")
        print(f"{'='*70}")
        
        created_examples = []
        
        for i, match in enumerate(matches, 1):
            example_id = f"example_{i:03d}"
            
            # Check if already exists
            example_dir = self.examples_dir / example_id
            if example_dir.exists():
                print(f"\n   ⚠️  {example_id} already exists, skipping...")
                continue
            
            print(f"\n   Creating {example_id}...")
            
            example = await self._create_example_from_match(
                example_id=example_id,
                match=match
            )
            
            if example:
                created_examples.append(example)
        
        # Summary
        print(f"\n{'='*70}")
        print(f"✅ PREPARATION COMPLETE")
        print(f"{'='*70}")
        print(f"\n   Total examples: {len(created_examples)}")
        
        if len(created_examples) >= 5:
            print(f"\n   🎉 You have {len(created_examples)} examples!")
            print(f"\n   Next step:")
            print(f"   python analyze_restructures_v1.py")
        else:
            print(f"\n   ⚠️  Need {5 - len(created_examples)} more examples")
            print(f"   Add more videos to the folder and run again")
        
        return created_examples
    
    def _find_best_longform(
        self,
        clip: Dict,
        longforms: List[Dict],
        min_similarity: float = 0.20  # Lowered to catch more matches
    ) -> Dict:
        """
        Find which longform this clip came from
        
        Uses text matching to find segments
        """
        
        best_match = None
        best_score = 0
        
        for longform in longforms:
            # Count how many clip segments match longform segments
            matched_count = 0
            total_similarity = 0
            
            for clip_seg in clip['segments']:
                clip_text = clip_seg['text'].lower()
                clip_words = set(clip_text.split())
                
                # Find best matching segment in longform
                for long_seg in longform['segments']:
                    long_text = long_seg['text'].lower()
                    long_words = set(long_text.split())
                    
                    # Jaccard similarity
                    intersection = len(clip_words & long_words)
                    union = len(clip_words | long_words)
                    jaccard = intersection / union if union > 0 else 0
                    
                    # ALSO check substring match (more lenient for exact clips)
                    clip_text_clean = ''.join(clip_text.split()).lower()
                    long_text_clean = ''.join(long_text.split()).lower()
                    substring_match = clip_text_clean in long_text_clean if len(clip_text_clean) > 15 else False
                    
                    # Match if EITHER good Jaccard OR substring found
                    if jaccard > 0.4 or substring_match:  # Lowered threshold
                        matched_count += 1
                        # Boost similarity for substring matches
                        total_similarity += max(jaccard, 0.7 if substring_match else 0)
                        break  # Found match for this clip segment
            
            # Calculate overall match score
            if matched_count > 0:
                avg_similarity = total_similarity / matched_count
                coverage = matched_count / len(clip['segments'])
                score = avg_similarity * coverage
                
                if score > best_score:
                    best_score = score
                    best_match = {
                        'longform': longform,
                        'similarity': avg_similarity,
                        'matched_count': matched_count,
                        'coverage': coverage
                    }
        
        # Return only if good enough match
        if best_match and best_match['similarity'] >= min_similarity:
            return best_match
        
        return None
    
    async def _create_example_from_match(
        self,
        example_id: str,
        match: Dict
    ) -> Dict:
        """Create example directory from match"""
        
        example_dir = self.examples_dir / example_id
        example_dir.mkdir(exist_ok=True)
        
        longform = match['longform']
        clip = match['clip']
        
        # Copy videos
        longform_dest = example_dir / "longform.mp4"
        clip_dest = example_dir / "viral_clip.mp4"
        
        shutil.copy2(longform['file'], longform_dest)
        shutil.copy2(clip['file'], clip_dest)
        
        # Copy transcripts
        longform_transcript_dest = example_dir / "longform_transcript.json"
        clip_transcript_dest = example_dir / "viral_clip_transcript.json"
        
        shutil.copy2(longform['transcript_path'], longform_transcript_dest)
        shutil.copy2(clip['transcript_path'], clip_transcript_dest)
        
        # Create metadata
        metadata = {
            'example_id': example_id,
            'created_at': datetime.now().isoformat(),
            'source_files': {
                'longform_original': str(longform['file']),
                'clip_original': str(clip['file'])
            },
            'longform': {
                'video': str(longform_dest),
                'transcript': str(longform_transcript_dest),
                'duration': longform['duration'],
                'segments': longform['segment_count']
            },
            'viral_clip': {
                'video': str(clip_dest),
                'transcript': str(clip_transcript_dest),
                'duration': clip['duration'],
                'segments': clip['segment_count'],
                'views': None,  # TODO: Ask user for views
                'vir_score': None
            },
            'match_quality': {
                'similarity': match['similarity'],
                'matched_segments': match['matched_segments'],
                'total_clip_segments': len(clip['segments'])
            },
            'status': 'ready'
        }
        
        metadata_file = example_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"      ✅ Created: {example_id}")
        print(f"         Similarity: {match['similarity']:.0%}")
        
        return metadata


async def main():
    """Main entry point"""
    
    prep = SmartRestructurePrep()
    
    # Use the folder you specified
    source_folder = "/Users/jervinquisada/custom-clip-finder/data/training/Longform and Clips"
    
    print(f"\n{'='*70}")
    print(f"🤖 SMART RESTRUCTURE DATA PREP")
    print(f"{'='*70}")
    print(f"\nThis will:")
    print(f"  1. Scan all videos in folder")
    print(f"  2. Transcribe each one (cached if already done)")
    print(f"  3. Auto-detect longforms vs clips")
    print(f"  4. Match clips to their source longforms")
    print(f"  5. Create organized training examples")
    
    response = input(f"\nReady to scan {source_folder}? (y/n): ").strip().lower()
    
    if response != 'y':
        print("Cancelled.")
        return
    
    # Scan and prepare
    await prep.scan_and_prepare(source_folder)


if __name__ == '__main__':
    asyncio.run(main())
