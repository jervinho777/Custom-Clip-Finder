#!/usr/bin/env python3
"""
FAST WITH CACHE - 2-stage pipeline with caching

First run: ~12 min, $4
Re-runs: ~3 min, $1 (cached!)
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict
import sys
import hashlib

sys.path.insert(0, str(Path(__file__).parent))

from test_2stage_fast import FastPipelineTest


class CachedFastPipeline(FastPipelineTest):
    """2-stage pipeline with comprehensive caching"""
    
    def __init__(self):
        super().__init__()
        self.cache_dir = Path("data/cache/pipeline")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load caches
        self.story_cache = self._load_cache("story_cache.json")
        self.moments_cache = self._load_cache("moments_cache.json")
        self.prescores_cache = self._load_cache("prescores_cache.json")
        self.godmode_cache = self._load_cache("godmode_cache.json")
    
    def _load_cache(self, filename: str) -> Dict:
        """Load cache file"""
        cache_file = self.cache_dir / filename
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return {}
    
    def _save_cache(self, filename: str, data: Dict):
        """Save cache file"""
        cache_file = self.cache_dir / filename
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _get_video_hash(self, video_path: str) -> str:
        """Get hash for video"""
        return hashlib.md5(str(video_path).encode()).hexdigest()[:8]
    
    def _get_moment_hash(self, moment: Dict) -> str:
        """Get hash for moment based on content"""
        start = moment.get('start', 0)
        end = moment.get('end', 0)
        segments = moment.get('segments', [])
        text = ''.join([s.get('text', '')[:100] for s in segments[:3]])
        content = f"{start:.1f}_{end:.1f}_{text}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    async def run_cached_pipeline(self, video_path: str, top_n: int = 5):
        """Run pipeline with caching"""
        
        print(f"\n{'='*70}")
        print(f"⚡ 2-STAGE CACHED PIPELINE")
        print(f"{'='*70}")
        
        video_hash = self._get_video_hash(video_path)
        
        story_cached = f"{video_hash}_story" in self.story_cache
        moments_cached = video_hash in self.moments_cache
        
        print(f"\n   Video: {Path(video_path).name}")
        print(f"   💾 Cache Status:")
        print(f"      Story: {'HIT ✅' if story_cached else 'MISS 🆕'}")
        print(f"      Moments: {'HIT ✅' if moments_cached else 'MISS 🆕'}")
        
        video_file = Path(video_path)
        transcript_file = Path(f"data/cache/transcripts/{video_file.stem}_transcript.json")
        
        with open(transcript_file) as f:
            data = json.load(f)
            segments = data['segments']
        
        print(f"   ✅ Loaded {len(segments)} segments")
        
        # STAGE 0: Story (cached)
        print(f"\n{'='*70}")
        print(f"STAGE 0: STORY ANALYSIS")
        print(f"{'='*70}")
        
        story_key = f"{video_hash}_story"
        
        if story_cached:
            print(f"   ✅ Using cached story")
            story = self.story_cache[story_key]
        else:
            print(f"   🆕 Analyzing story...")
            story = await self.v4._analyze_story_ensemble(segments)
            self.story_cache[story_key] = story
            self._save_cache("story_cache.json", self.story_cache)
        
        # STAGE 1: Moments (cached)
        print(f"\n{'='*70}")
        print(f"STAGE 1: FIND MOMENTS")
        print(f"{'='*70}")
        
        if moments_cached:
            print(f"   ✅ Using cached moments")
            moments = self.moments_cache[video_hash]
        else:
            print(f"   🆕 Finding moments...")
            moments = await self.v4._find_moments_with_consensus(segments, story)
            self.moments_cache[video_hash] = moments
            self._save_cache("moments_cache.json", self.moments_cache)
        
        print(f"\n   Found: {len(moments)} moments")
        
        # STAGE 1.5: RESTRUCTURE ALL MOMENTS (NEW!)
        print(f"\n{'='*70}")
        print(f"STAGE 1.5: RESTRUCTURE MOMENTS")
        print(f"{'='*70}")
        
        print(f"\n   🔧 Restructuring {len(moments)} moments with learnings...")
        
        restructured_moments = []
        restructure_cached = 0
        restructure_new = 0
        
        for i, moment in enumerate(moments, 1):
            # Create hash for caching
            moment_hash = self._get_moment_hash(moment)
            restructure_key = f"restructure_{moment_hash}"
            
            # Check cache
            if restructure_key in self.moments_cache:
                restructured = self.moments_cache[restructure_key]
                restructure_cached += 1
                status = "💾"
            else:
                # Restructure with AI + learnings
                restructured = await self.v4._restructure_with_review(
                    moment=moment,
                    segments=segments,
                    story_structure=story
                )
                
                # If restructure fails, keep original
                if restructured is None:
                    restructured = moment
                
                # Cache the result
                self.moments_cache[restructure_key] = restructured
                self._save_cache("moments_cache.json", self.moments_cache)
                restructure_new += 1
                status = "🆕"
            
            restructured_moments.append(restructured)
            print(f"   {status} {i}/{len(moments)}: Restructured", end='\r')
        
        moments = restructured_moments
        
        print(f"\n   ✅ Restructured {len(moments)} moments")
        print(f"   💾 Cache: {restructure_cached} cached, {restructure_new} new")
        
        # STAGE 2: Pre-screen (cached)
        print(f"\n{'='*70}")
        print(f"STAGE 2: PRE-SCREENING")
        print(f"{'='*70}")
        
        scored_moments = []
        cached_count = 0
        new_count = 0
        
        for i, moment in enumerate(moments, 1):
            moment_hash = self._get_moment_hash(moment)
            
            if moment_hash in self.prescores_cache:
                score_result = self.prescores_cache[moment_hash]
                cached_count += 1
                status = "💾"
            else:
                score_result = await self.quick_pre_score_opus(moment, segments)
                self.prescores_cache[moment_hash] = score_result
                self._save_cache("prescores_cache.json", self.prescores_cache)
                new_count += 1
                status = "🆕"
            
            scored_moments.append({
                **moment,
                'pre_score': score_result['score'],
                'pre_reason': score_result['reason']
            })
            
            print(f"   {status} {i}/{len(moments)}: {score_result['score']}/100", end='\r')
        
        print(f"\n   Pre-scored: {len(scored_moments)} ({cached_count} cached, {new_count} new)")
        
        scored_moments.sort(key=lambda m: m['pre_score'], reverse=True)
        top_moments = [m for m in scored_moments if m['pre_score'] >= 70][:top_n]
        
        if not top_moments:
            top_moments = scored_moments[:top_n]
        
        print(f"   Selected: {len(top_moments)} candidates")
        
        # STAGE 3: Godmode (cached)
        print(f"\n{'='*70}")
        print(f"STAGE 3: GODMODE")
        print(f"{'='*70}")
        
        results = []
        cached_eval = 0
        new_eval = 0
        
        for i, moment in enumerate(top_moments, 1):
            moment_hash = self._get_moment_hash(moment)
            
            if moment_hash in self.godmode_cache:
                result = self.godmode_cache[moment_hash]
                cached_eval += 1
                status = "💾"
            else:
                print(f"\n   🆕 {i}/{len(top_moments)}: Evaluating...")
                
                moment_segments = [
                    seg for seg in segments
                    if moment['start'] <= seg.get('start', 0) < moment['end']
                ]
                
                clip = {
                    'start': moment['start'],
                    'end': moment['end'],
                    'segments': moment_segments,
                    'video_path': str(video_path),
                    'pre_score': moment['pre_score'],
                    'structure': {
                        'segments': moment_segments
                    }
                }
                
                story_eval = {'storylines': [], 'standalone_moments': []}
                result = await self.v4._evaluate_quality_debate(clip, story_eval)
                
                self.godmode_cache[moment_hash] = result
                self._save_cache("godmode_cache.json", self.godmode_cache)
                new_eval += 1
                status = "🆕"
            
            results.append({
                'clip': moment,
                'score': result.get('total_score', result.get('score', 0)),
                'tier': result.get('quality_tier', 'F'),
                'pre_score': moment['pre_score']
            })
            
            print(f"   {status} {i}/{len(top_moments)}: {moment['pre_score']}/100 → {result.get('total_score', result.get('score', 0))}/50")
        
        # Summary
        passing = [r for r in results if r['score'] >= 40]
        
        print(f"\n{'='*70}")
        print(f"📊 RESULTS")
        print(f"{'='*70}")
        
        print(f"\n   ✅ Passed (40+): {len(passing)}")
        
        if passing:
            for i, r in enumerate(passing, 1):
                print(f"      {i}. {r['pre_score']}/100 → {r['score']}/50 ({r['tier']})")
        
        # Cache stats
        total = 2 + len(moments) + len(top_moments)
        cached = (1 if story_cached else 0) + (1 if moments_cached else 0) + cached_count + cached_eval
        rate = cached / total * 100 if total > 0 else 0
        
        print(f"\n   💾 Cache: {rate:.0f}%")
        
        if rate > 80:
            print(f"   ⚡ Next run: ~3 min, $1")
        elif rate > 50:
            print(f"   ✅ Next run: ~6 min, $2")
        else:
            print(f"   🆕 First run: ~12 min, $4")
            print(f"   🔄 Next run will be faster!")
        
        return results


async def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "data/uploads/Dieter Lange.mp4"
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    tester = CachedFastPipeline()
    await tester.run_cached_pipeline(video, top_n)


if __name__ == '__main__':
    asyncio.run(main())

