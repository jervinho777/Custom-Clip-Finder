#!/usr/bin/env python3
"""
🎬 CREATE CLIPS V4 - Integrated 5-AI Ensemble Pipeline
Complete System: Story-First + Multi-AI Consensus

Pipeline:
STEP 0: Story Analysis (Ensemble Consensus)
STEP 1: Find Moments (5-AI Parallel Vote)
STEP 2: Restructure (Hybrid: Lead + Review)
STEP 2.5: Quality Eval (5-AI Debate)
STEP 3: Variations (Fast Models)
EXPORT: MP4 + XML + JSON
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import base systems
try:
    from create_clips_v2 import CreateClipsV2
except ImportError:
    print("⚠️  Warning: create_clips_v2.py not found")
    CreateClipsV2 = None

try:
    from create_clips_v3_ensemble import PremiumAIEnsemble, ConsensusEngine
except ImportError:
    print("⚠️  Warning: create_clips_v3_ensemble.py not found")
    PremiumAIEnsemble = None
    ConsensusEngine = None

try:
    from master_learnings_v2 import get_learnings_for_prompt
except ImportError:
    print("⚠️  Warning: master_learnings_v2.py not found")
    get_learnings_for_prompt = None


class CreateClipsV4Integrated:
    """
    V4: Integrated 5-AI Ensemble System
    
    Combines:
    - Story-First approach (V2)
    - Multi-AI Consensus (V3)
    - Quality-First philosophy
    """
    
    # Quality thresholds (moderate lowering)
    QUALITY_PASS_THRESHOLD = 18  # Minimum score to pass (was 20)
    
    def __init__(self):
        print("\n" + "="*70)
        print("🎬 CREATE CLIPS V4 - INTEGRATED 5-AI ENSEMBLE")
        print("="*70)
        
        # Initialize base system (for video processing, export, etc.)
        if CreateClipsV2:
            self.base_system = CreateClipsV2()
            print("   ✅ Base System: Ready")
        else:
            self.base_system = None
            print("   ⚠️  Base System: Not available")
        
        # Initialize AI ensemble
        if PremiumAIEnsemble and ConsensusEngine:
            self.ensemble = PremiumAIEnsemble()
            self.consensus = ConsensusEngine(self.ensemble)
            print("   ✅ AI Ensemble: Ready")
            print("   ✅ Consensus Engine: Ready")
        else:
            self.ensemble = None
            self.consensus = None
            print("   ⚠️  AI Ensemble: Not available")
        
        # Set up directories
        self.data_dir = Path("data")
        self.transcript_dir = self.data_dir / "cache" / "transcripts"
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70 + "\n")
    
    def _get_quality_tier(self, score: int) -> str:
        """
        Convert score to quality tier
        
        Moderate thresholds (lowered by 2-3 points)
        """
        if score >= 38:  # Was 40 (A-tier slightly easier)
            return 'A'
        elif score >= 28:  # Was 30 (B-tier moderate)
            return 'B'
        elif score >= 18:  # Was 20 (C-tier moderate)
            return 'C'
        else:
            return 'D'
    
    def _extract_tier_letter(self, tier_string: str) -> str:
        """
        Extract base letter from tier string
        
        Examples:
        - 'A' → 'A'
        - 'A+' → 'A'
        - 'B-' → 'B'
        - 'C+' → 'C'
        """
        if not tier_string or not isinstance(tier_string, str):
            return 'D'
        
        # Get first character and uppercase
        tier_letter = tier_string.strip()[0].upper()
        
        # Validate it's A-D
        if tier_letter in ['A', 'B', 'C', 'D']:
            return tier_letter
        
        return 'D'  # Default to D if invalid
    
    async def extract_clips(self, segments: List[Dict], video_path: str) -> Dict:
        """
        Complete integrated pipeline with 5-AI ensemble
        
        Returns:
            {
                'story_structure': {...},
                'clips': [{...}],
                'consensus_data': {...},
                'stats': {...}
            }
        """
        
        print("\n" + "="*70)
        print("🎬 INTEGRATED PIPELINE - START")
        print("="*70)
        
        # STEP 0: Story Analysis with Ensemble
        print("\n📖 STEP 0: Story Analysis (Ensemble Consensus)")
        story_structure = await self._analyze_story_ensemble(segments)
        
        # STEP 1: Find Moments with 5-AI Consensus
        print("\n📊 STEP 1: Find Moments (5-AI Parallel Vote)")
        moments = await self._find_moments_ensemble(segments, story_structure)
        
        print(f"\n   ✅ Found {len(moments)} consensus moments")
        
        # STEP 1.5: Quick Quality Check - Skip restructure for high-scoring clips
        print("\n⭐ STEP 1.5: Quick Quality Check (Pre-Restructure)")
        high_quality_clips = []
        needs_restructure = []
        
        for i, moment in enumerate(moments[:10], 1):  # Limit to 10 for now
            print(f"\n   📊 Quick evaluation: Moment {i}/{min(len(moments), 10)}")
            
            # Extract segments for this moment
            moment_start = moment.get('start', 0)
            moment_end = moment.get('end', 0)
            moment_segments = [
                seg for seg in segments
                if moment_start <= seg.get('start', 0) < moment_end
            ]
            
            # Convert moment to clip format for evaluation
            moment_clip = {
                'start': moment_start,
                'end': moment_end,
                'segments': moment_segments,
                'clip_id': f"moment_{i}",
                'structure': {
                    'segments': moment_segments
                }
            }
            
            # Quick quality check
            quality = await self._evaluate_quality_debate(moment_clip, story_structure)
            score = quality.get('total_score', 0)
            
            if score >= 40:  # Already excellent - skip restructure!
                print(f"      ✅ Score: {score}/50 - Preserving original structure")
                moment_clip['quality'] = quality
                moment_clip['preserved_original'] = True
                high_quality_clips.append(moment_clip)
            else:
                print(f"      🔧 Score: {score}/50 - Needs restructure")
                needs_restructure.append((moment, quality))
        
        print(f"\n   ✅ High-quality (preserved): {len(high_quality_clips)}")
        print(f"   🔧 Needs restructure: {len(needs_restructure)}")
        
        # STEP 2: Restructure with Lead + Review (only for clips that need it)
        restructured = []
        
        if needs_restructure:
            print("\n🔄 STEP 2: Restructure (Hybrid: Lead + Review)")
            for i, (moment, pre_quality) in enumerate(needs_restructure, 1):
                print(f"\n   📌 Restructuring {i}/{len(needs_restructure)}")
                clip = await self._restructure_with_review(moment, segments, story_structure)
                if clip:
                    # Re-evaluate after restructure
                    quality = await self._evaluate_quality_debate(clip, story_structure)
                    clip['quality'] = quality
                    clip['preserved_original'] = False
                    restructured.append(clip)
            
            print(f"\n   ✅ Restructured {len(restructured)} clips")
        else:
            print("\n   ✅ All clips already high quality - skipping restructure!")
        
        # STEP 2.5: Combine high-quality preserved + restructured clips
        print("\n⭐ STEP 2.5: Final Quality Check")
        quality_passed = []
        
        # Add preserved high-quality clips
        quality_passed.extend(high_quality_clips)
        
        # Add restructured clips (already evaluated)
        quality_passed.extend(restructured)
        
        # Final filter by tier
        final_passed = []
        for clip in quality_passed:
            quality = clip.get('quality', {})
            tier_string = quality.get('quality_tier', 'D')
            tier_letter = self._extract_tier_letter(tier_string)
            
            if tier_letter in ['A', 'B', 'C']:
                final_passed.append(clip)
                preserved = "✅ PRESERVED" if clip.get('preserved_original') else "🔄 RESTRUCTURED"
                print(f"      {preserved} - {tier_string} ({tier_letter})")
            else:
                print(f"      ❌ {tier_string} ({tier_letter}) - Rejected")
        
        quality_passed = final_passed
        print(f"\n   ✅ {len(quality_passed)} clips passed quality gate")
        
        # STEP 3: Create Variations (Fast Models)
        print("\n🎨 STEP 3: Create Variations (Fast Models)")
        all_clips_with_versions = []
        
        for clip in quality_passed:
            # Determine variations based on quality tier
            tier_string = clip.get('quality', {}).get('quality_tier', 'C')
            tier_letter = self._extract_tier_letter(tier_string)
            max_variations = 3 if tier_letter == 'A' else 2 if tier_letter == 'B' else 1
            
            if self.base_system:
                variations = self.base_system._create_variations(
                    clip, 
                    segments
                )
            else:
                # Fallback: just use original
                variations = [{
                    'version_id': f"{clip.get('clip_id', 'clip')}_original",
                    'version_name': 'Original',
                    'variation_type': 'original',
                    'structure': clip.get('structure', {})
                }]
            
            all_clips_with_versions.append({
                'clip': clip,
                'versions': variations[:max_variations]
            })
        
        total_versions = sum(len(c['versions']) for c in all_clips_with_versions)
        print(f"\n   ✅ Created {total_versions} total versions")
        
        # Summary
        print("\n" + "="*70)
        print("📊 PIPELINE SUMMARY")
        print("="*70)
        print(f"   Story Analysis: Consensus-based")
        print(f"   Moments Found: {len(moments)}")
        print(f"   High-Quality Preserved: {len(high_quality_clips)}")
        print(f"   Clips Restructured: {len(restructured)}")
        print(f"   Quality Passed: {len(quality_passed)}")
        print(f"   Total Versions: {total_versions}")
        
        # Print costs
        if self.ensemble:
            self.ensemble.print_usage_stats()
        
        # Collect validation stats from quality evaluations
        validations = []
        for clip_data in all_clips_with_versions:
            clip = clip_data.get('clip', {})
            quality = clip.get('quality', {})
            validation = quality.get('validation', {})
            if validation:
                validations.append(validation)
        
        # Print validation statistics
        if validations:
            self._print_validation_stats(validations)
        
        return {
            'story_structure': story_structure,
            'moments': moments,
            'clips': all_clips_with_versions,
            'stats': {
                'moments_found': len(moments),
                'clips_restructured': len(restructured),
                'quality_passed': len(quality_passed),
                'total_versions': total_versions,
                'ai_consensus_used': True,
                'validation_stats': {
                    'total_validated': len(validations),
                    'avg_confidence': sum(v.get('confidence', 0) for v in validations) / len(validations) if validations else 0,
                    'learnings_applied_count': sum(1 for v in validations if v.get('learnings_applied'))
                }
            }
        }
    
    async def _analyze_story_ensemble(self, segments: List[Dict]) -> Dict:
        """
        Story analysis using ensemble consensus WITH LEARNINGS
        
        Uses parallel vote for fast story understanding
        Now includes learned patterns and algorithm context
        """
        
        if not self.base_system or not self.consensus:
            # Fallback
            if self.base_system:
                return self.base_system._analyze_story_structure(segments)
            return {'storylines': [], 'standalone_moments': []}
        
        # GET LEARNINGS (with graceful fallback)
        learnings_prompt = self._get_learnings_safely()
        learnings_section = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{learnings_prompt}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        # Format transcript
        transcript = self.base_system._format_segments(segments, max_chars=50000)
        duration = segments[-1]['end'] if segments else 0
        
        prompt = f"""{learnings_section}

# 📹 VIDEO TRANSCRIPT ({duration:.0f} seconds):

{transcript[:30000]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 📖 STORY ANALYSE BASIEREND AUF LEARNINGS:

Analysiere die Story-Struktur BASIEREND auf:
1. Den gelernten Viral Patterns (oben)
2. Dem Algorithmus-Kontext (Watchtime-Maximierung)
3. Den {self._get_clips_analyzed_count()}+ analysierten Beispielen

Identifiziere:

## 1. STORYLINES (verbundene Narrative)
- Welche Storylines aus den Learnings passen?
- Folgt es winning structures?
- Welche Hook-Types aus den Patterns sind erkennbar?
- Algorithmus-Impact: Watchtime-Potential pro Storyline?

## 2. STANDALONE MOMENTS (selbsterklärend)
- Welche Momente können als Clip funktionieren OHNE Context?
- Nutzen sie winning hook types?
- Algorithmus-Impact: Hohe Completion-Rate möglich?

## 3. DEPENDENCIES (was braucht Context?)
- Welche Segmente MÜSSEN zusammenbleiben?
- Was passiert wenn man sie trennt (Algorithmus-Impact)?
- Welche Patterns aus Learnings erfordern Context?

Antworte in JSON:
{{
  "storylines": [
    {{
      "storyline_id": "story_1",
      "topic": "...",
      "segments": [
        {{"start": X, "end": Y, "role": "...", "key_elements": [...]}}
      ],
      "can_standalone": true/false,
      "requires_context": true/false,
      "learned_patterns": ["Pattern 1", "Pattern 2"],
      "watchtime_potential": "high/medium/low",
      "algorithm_assessment": "Warum Algorithmus diese Storyline pushen würde"
    }}
  ],
  "standalone_moments": [
    {{
      "start": X,
      "end": Y,
      "topic": "...",
      "why_standalone": "...",
      "hook_type": "question/statement/story/number",
      "learned_patterns": ["Pattern 1"],
      "watchtime_potential": "high/medium/low"
    }}
  ],
  "analysis": {{
    "storylines_count": X,
    "standalone_count": Y,
    "learned_patterns_found": ["Pattern 1", "Pattern 2"],
    "algorithm_insights": "Warum diese Struktur Watchtime maximiert"
  }}
}}
"""
        
        system = f"Du bist ein Story-Analyst trainiert auf {self._get_clips_analyzed_count()} viralen Clips mit Algorithmus-Verständnis."
        
        # Use parallel vote for fast analysis
        result = await self.consensus.build_consensus(
            prompt=prompt,
            system=system,
            strategy='parallel_vote'
        )
        
        print(f"\n   Consensus Confidence: {result.get('confidence', 0):.0%}")
        
        # Parse consensus result
        try:
            consensus_text = result.get('consensus', '')
            # Try to extract JSON from consensus
            if '{' in consensus_text:
                json_start = consensus_text.find('{')
                json_end = consensus_text.rfind('}') + 1
                json_str = consensus_text[json_start:json_end]
                story_structure = json.loads(json_str)
                return story_structure
        except Exception as e:
            print(f"   ⚠️  Parse error: {e}")
        
        # Fallback to base system
        print("   ⚠️  Falling back to base system for story analysis")
        return self.base_system._analyze_story_structure(segments)
    
    async def _find_moments_ensemble(self, segments: List[Dict], 
                                     story_structure: Dict) -> List[Dict]:
        """
        Find moments using 5-AI consensus
        
        Each AI finds moments, then consensus builds final list
        """
        
        if not self.base_system:
            return []
        
        # Use chunked approach for long videos
        total_duration = segments[-1]['end'] if segments else 0
        
        if total_duration > 600:  # > 10 minutes
            print(f"   📦 Long video - using chunked processing")
            return await self._find_moments_chunked_ensemble(segments, story_structure)
        else:
            return await self._find_moments_single_ensemble(segments, story_structure)
    
    async def _find_moments_single_ensemble(self, segments: List[Dict],
                                           story_structure: Dict) -> List[Dict]:
        """
        Find moments in single pass with ensemble
        """
        
        if not self.base_system or not self.consensus:
            # Fallback
            if self.base_system:
                return self.base_system._find_moments_single_pass(segments)
            return []
        
        transcript = self.base_system._format_segments(segments, max_chars=20000)
        
        prompt = f"""Finde die besten Momente für Short-Form Clips.

TRANSCRIPT:
{transcript}

Finde 10-15 starke Momente.

Antworte in JSON:
{{
  "moments": [
    {{
      "id": 1,
      "start": X,
      "end": Y,
      "topic": "...",
      "strength": "high/medium",
      "reason": "..."
    }}
  ]
}}
"""
        
        system = "Du bist ein Experte für virale Short-Form Videos."
        
        # Use parallel vote
        result = await self.consensus.build_consensus(
            prompt=prompt,
            system=system,
            strategy='parallel_vote'
        )
        
        print(f"   Consensus Confidence: {result.get('confidence', 0):.0%}")
        
        # Parse moments
        try:
            consensus_text = result.get('consensus', '')
            # Try to extract JSON from consensus
            if '{' in consensus_text:
                json_start = consensus_text.find('{')
                json_end = consensus_text.rfind('}') + 1
                json_str = consensus_text[json_start:json_end]
                parsed = json.loads(json_str)
                moments = parsed.get('moments', [])
                
                # Add consensus metadata
                for moment in moments:
                    moment['found_by'] = 'ensemble_consensus'
                    moment['consensus_confidence'] = result.get('confidence', 0)
                
                return moments
        except Exception as e:
            print(f"   ⚠️  Parse error: {e}")
        
        # Fallback
        print("   ⚠️  Parse error, falling back to base system")
        return self.base_system._find_moments_single_pass(segments)
    
    async def _find_moments_chunked_ensemble(self, segments: List[Dict],
                                            story_structure: Dict) -> List[Dict]:
        """
        Find moments in chunks using ensemble
        """
        
        if not self.base_system:
            return []
        
        chunk_size = 300  # 5 minutes
        total_duration = segments[-1]['end'] if segments else 0
        
        all_moments = []
        chunk_num = 1
        chunk_start = 0
        
        while chunk_start < total_duration:
            chunk_end = min(chunk_start + chunk_size, total_duration)
            
            print(f"\n   🔍 Chunk {chunk_num}: {chunk_start/60:.1f}-{chunk_end/60:.1f} min")
            
            # Get segments for chunk
            chunk_segments = [
                s for s in segments
                if chunk_start <= s['start'] < chunk_end
            ]
            
            if chunk_segments:
                # Find moments in chunk (using base system for speed)
                try:
                    chunk_moments = self.base_system._find_moments_in_chunk(
                        chunk_segments,
                        chunk_start,
                        chunk_end,
                        chunk_num
                    )
                    
                    print(f"      ✅ Found {len(chunk_moments)} moments")
                    all_moments.extend(chunk_moments)
                except Exception as e:
                    print(f"      ⚠️  Error in chunk: {e}")
                    # Fallback: use single pass for this chunk
                    try:
                        fallback_moments = self.base_system._find_moments_single_pass(chunk_segments)
                        all_moments.extend(fallback_moments)
                    except:
                        pass
            
            chunk_start = chunk_end
            chunk_num += 1
        
        return all_moments
    
    def _find_exact_hook_location(self, segments: List[Dict], hook_text: str, approximate_time: float) -> float:
        """
        Find exact location of hook text in segments
        
        Args:
            segments: All video segments
            hook_text: Text to find (from AI analysis)
            approximate_time: AI's approximate timestamp
        
        Returns:
            Exact start time of hook
        """
        # Clean hook text for matching
        hook_clean = hook_text.lower().strip()
        hook_words = hook_clean.split()[:5]  # First 5 words
        
        # Search within ±30s window of approximate time
        search_start = max(0, approximate_time - 30)
        search_end = approximate_time + 30
        
        # Find segments in search window
        candidates = [
            seg for seg in segments
            if search_start <= seg.get('start', 0) <= search_end
        ]
        
        # Try to find exact match
        for seg in candidates:
            seg_text = seg.get('text', '').lower()
            
            # Check if hook words are in this segment
            if any(word in seg_text for word in hook_words if len(word) > 3):
                return seg['start']
        
        # Fallback: find closest segment to approximate time
        if candidates:
            closest = min(candidates, key=lambda s: abs(s['start'] - approximate_time))
            return closest['start']
        
        # Last resort: use approximate time
        return approximate_time
    
    def _calculate_moment_end(self, segments: List[Dict], start_time: float, 
                              min_duration: float = 35, max_duration: float = 70) -> float:
        """
        Calculate appropriate end time for moment to include complete story
        
        Args:
            segments: All video segments
            start_time: Moment start time
            min_duration: Minimum clip length (default 35s)
            max_duration: Maximum clip length (default 70s)
        
        Returns:
            End time that includes complete thought/story
        """
        # Get segments from start time onward
        future_segments = [
            seg for seg in segments
            if seg.get('start', 0) >= start_time
        ]
        
        if not future_segments:
            return start_time + min_duration
        
        current_duration = 0
        end_time = start_time + min_duration
        
        for seg in future_segments:
            current_duration = seg.get('end', start_time) - start_time
            
            # Check if we've reached minimum and have complete sentence
            if current_duration >= min_duration:
                text = seg.get('text', '')
                
                # Look for natural ending points
                if any(text.strip().endswith(end) for end in ['.', '!', '?', '...']):
                    end_time = seg.get('end', start_time)
                    break
            
            # Don't exceed maximum
            if current_duration >= max_duration:
                end_time = seg.get('end', start_time)
                break
            
            # Update potential end time
            end_time = seg.get('end', start_time)
        
        # Ensure minimum duration
        if end_time - start_time < min_duration:
            end_time = start_time + min_duration
        
        return end_time
    
    def _extract_moment_segments(self, segments: List[Dict], start_time: float, 
                                 end_time: float) -> List[Dict]:
        """
        Extract segments for a moment with proper boundaries
        
        Args:
            segments: All video segments
            start_time: Moment start
            end_time: Moment end
        
        Returns:
            List of segments within moment boundaries
        """
        moment_segments = []
        
        for seg in segments:
            seg_start = seg.get('start', 0)
            seg_end = seg.get('end', 0)
            
            # Segment overlaps with moment
            if seg_start < end_time and seg_end > start_time:
                moment_segments.append(seg)
        
        return moment_segments
    
    def _validate_moment(self, moment: Dict) -> bool:
        """
        Validate that moment has proper structure (WITH DEBUG)
        
        Args:
            moment: Moment dict to validate
        
        Returns:
            True if valid, False otherwise
        """
        
        # Must have segments
        if not moment.get('segments'):
            return False
        
        # Must have reasonable duration
        duration = moment.get('end', 0) - moment.get('start', 0)
        if duration < 10 or duration > 120:
            return False
        
        # Must have text content
        total_text = ''.join([seg.get('text', '') for seg in moment['segments']])
        if len(total_text.strip()) < 50:  # At least 50 chars
            return False
        
        return True
    
    def _format_transcript_for_analysis(self, segments: List[Dict]) -> str:
        """Format segments for AI analysis"""
        if not self.base_system:
            return '\n'.join([f"[{s['start']:.1f}s] {s.get('text', '')}" for s in segments])
        return self.base_system._format_segments(segments, max_chars=20000)
    
    async def _find_moments_with_consensus(self, segments: List[Dict], 
                                          story: Dict) -> List[Dict]:
        """
        Find viral moments using AI consensus (FIXED VERSION WITH DEBUG)
        """
        
        print(f"\n📊 STEP 1: Find Moments (5-AI Parallel Vote)")
        
        # Check if long video needs chunking
        video_duration = segments[-1]['end'] if segments else 0
        
        if video_duration > 300:  # 5+ minutes
            print(f"   📦 Long video - using chunked processing")
            moments = await self._find_moments_chunked_fixed(segments, story)
        else:
            # Short video - process all at once
            result = await self._find_moments_chunk_fixed(
                segments=segments,
                story=story,
                chunk_start=0,
                chunk_end=video_duration
            )
            moments = result.get('moments', [])
        
        print(f"\n   ✅ Found {len(moments)} consensus moments")
        
        # DEBUG SUMMARY
        print(f"\n   {'='*70}")
        print(f"   🔍 DEBUG: MOMENT EXTRACTION SUMMARY")
        print(f"   {'='*70}")
        
        valid_count = 0
        invalid_count = 0
        empty_segments = 0
        too_short = 0
        too_long = 0
        no_text = 0
        
        for m in moments:
            is_valid = self._validate_moment(m)
            
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                
                # Track why invalid
                if not m.get('segments'):
                    empty_segments += 1
                
                duration = m.get('end', 0) - m.get('start', 0)
                if duration < 10:
                    too_short += 1
                elif duration > 120:
                    too_long += 1
                
                if m.get('segments'):
                    text = ''.join([s.get('text', '') for s in m['segments']])
                    if len(text.strip()) < 50:
                        no_text += 1
        
        print(f"\n   ✅ Valid moments: {valid_count}")
        print(f"   ❌ Invalid moments: {invalid_count}")
        
        if invalid_count > 0:
            print(f"\n   ⚠️  Invalid breakdown:")
            if empty_segments > 0:
                print(f"      - No segments: {empty_segments}")
            if too_short > 0:
                print(f"      - Too short (<10s): {too_short}")
            if too_long > 0:
                print(f"      - Too long (>120s): {too_long}")
            if no_text > 0:
                print(f"      - No text (<50 chars): {no_text}")
        
        # Show first 5 moments in detail
        print(f"\n   🔍 DEBUG: First 5 moments:")
        for i, m in enumerate(moments[:5], 1):
            duration = m.get('end', 0) - m.get('start', 0)
            seg_count = len(m.get('segments', []))
            is_valid = self._validate_moment(m)
            
            status = "✅" if is_valid else "❌"
            
            print(f"\n   {status} Moment {i}:")
            print(f"      Start: {m.get('start', 0):.1f}s")
            print(f"      End: {m.get('end', 0):.1f}s")
            print(f"      Duration: {duration:.1f}s")
            print(f"      Segments: {seg_count}")
            
            if seg_count > 0:
                first_text = m['segments'][0].get('text', 'NO TEXT')[:60]
                print(f"      First text: {first_text}...")
                
                if not is_valid:
                    # Show why invalid
                    text = ''.join([s.get('text', '') for s in m['segments']])
                    if duration < 10:
                        print(f"      ⚠️  Too short: {duration:.1f}s < 10s")
                    elif duration > 120:
                        print(f"      ⚠️  Too long: {duration:.1f}s > 120s")
                    if len(text.strip()) < 50:
                        print(f"      ⚠️  Too little text: {len(text)} chars < 50")
            else:
                print(f"      ❌ NO SEGMENTS!")
        
        print(f"   {'='*70}\n")
        
        return moments
    
    async def _find_moments_chunked_fixed(self, segments: List[Dict], 
                                         story: Dict) -> List[Dict]:
        """
        Process long videos in chunks with FIXED extraction
        """
        video_duration = segments[-1]['end']
        chunk_size = 300  # 5 min chunks
        
        all_moments = []
        chunk_num = 1
        
        for chunk_start in range(0, int(video_duration), chunk_size):
            chunk_end = min(chunk_start + chunk_size, video_duration)
            
            print(f"\n   🔍 Chunk {chunk_num}: {chunk_start/60:.1f}-{chunk_end/60:.1f} min")
            
            result = await self._find_moments_chunk_fixed(
                segments=segments,
                story=story,
                chunk_start=chunk_start,
                chunk_end=chunk_end
            )
            
            moments = result.get('moments', [])
            
            print(f"      ✅ Found {len(moments)} moments")
            
            all_moments.extend(moments)
            chunk_num += 1
        
        print(f"\n   ✅ Found {len(all_moments)} consensus moments")
        
        return all_moments
    
    async def _find_moments_chunk_fixed(self, segments: List[Dict], story: Dict,
                                       chunk_start: float, chunk_end: float) -> Dict:
        """
        Find moments in a chunk with FIXED extraction logic (WITH DEBUG)
        """
        
        print(f"\n      🔍 DEBUG: Processing chunk {chunk_start/60:.1f}-{chunk_end/60:.1f} min")
        
        if not self.consensus:
            return {'moments': []}
        
        # Get segments for this chunk
        chunk_segments = [
            seg for seg in segments
            if chunk_start <= seg.get('start', 0) < chunk_end
        ]
        
        print(f"      🔍 DEBUG: Chunk has {len(chunk_segments)} segments")
        
        # Format for AI
        chunk_text = self._format_transcript_for_analysis(chunk_segments[:50])
        
        # AI prompt
        prompt = f"""
Analyze this transcript segment and identify potential VIRAL MOMENTS.

TRANSCRIPT ({chunk_end - chunk_start:.0f}s):
{chunk_text}

For each moment, specify:
1. Approximate timestamp (in seconds from chunk start)
2. Key hook phrase (exact text)
3. Viral pattern used
4. Why it's viral-worthy

Respond with JSON array of moments.
"""
        
        # Call AI consensus
        result = await self.consensus.build_consensus(
            prompt=prompt,
            system="You are a viral content expert finding moments.",
            strategy='parallel_vote'
        )
        
        consensus_text = result.get('consensus', '{}')
        
        # Parse AI response
        try:
            if '```json' in consensus_text:
                json_start = consensus_text.find('```json') + 7
                json_end = consensus_text.find('```', json_start)
                consensus_text = consensus_text[json_start:json_end]
            elif '[' in consensus_text and consensus_text.strip().startswith('['):
                # AI returned array directly
                json_start = consensus_text.find('[')
                json_end = consensus_text.rfind(']') + 1
                consensus_text = consensus_text[json_start:json_end]
            elif '{' in consensus_text:
                json_start = consensus_text.find('{')
                json_end = consensus_text.rfind('}') + 1
                consensus_text = consensus_text[json_start:json_end]
            
            data = json.loads(consensus_text)
            
            # Handle both list and dict formats
            if isinstance(data, list):
                # AI returned array directly: [{moment1}, {moment2}]
                ai_moments = data
                print(f"      🔍 DEBUG: AI returned list format ({len(data)} moments)")
            elif isinstance(data, dict):
                # AI returned dict: {moments: [...]}
                ai_moments = data.get('moments', [])
                print(f"      🔍 DEBUG: AI returned dict format ({len(ai_moments)} moments)")
            else:
                print(f"      🔍 DEBUG: Unexpected data type: {type(data)}")
                ai_moments = []
        
        except Exception as e:
            print(f"      🔍 DEBUG: JSON parse error: {e}")
            ai_moments = []
        
        # FIXED: Extract moments with proper timestamps
        extracted_moments = []
        
        for idx, ai_moment in enumerate(ai_moments, 1):
            # Get AI's approximate location
            approx_time = ai_moment.get('timestamp', 0) + chunk_start
            hook_text = ai_moment.get('hook_phrase', '')
            
            # FIXED: Find exact hook location
            exact_start = self._find_exact_hook_location(
                segments=segments,
                hook_text=hook_text,
                approximate_time=approx_time
            )
            
            # FIXED: Calculate proper end time
            exact_end = self._calculate_moment_end(
                segments=segments,
                start_time=exact_start,
                min_duration=35,
                max_duration=70
            )
            
            # FIXED: Extract segments with proper boundaries
            moment_segments = self._extract_moment_segments(
                segments=segments,
                start_time=exact_start,
                end_time=exact_end
            )
            
            # Create moment object
            moment = {
                'start': exact_start,
                'end': exact_end,
                'segments': moment_segments,
                'pattern': ai_moment.get('pattern', ''),
                'hook_phrase': hook_text,
                'ai_reasoning': ai_moment.get('reason', '')
            }
            
            # FIXED: Validate before adding
            is_valid = self._validate_moment(moment)
            
            if is_valid:
                extracted_moments.append(moment)
        
        print(f"      ✅ Extracted: {len(extracted_moments)}/{len(ai_moments)} moments valid")
        
        return {'moments': extracted_moments}
    
    async def _restructure_with_review(self, moment: Dict, segments: List[Dict],
                                      story_structure: Dict) -> Optional[Dict]:
        """
        Restructure with AI returning INDICES (preserves original segments with text)
        
        AI returns which segment indices to keep, we use original segments
        """
        
        if not self.base_system or not self.consensus:
            return None
        
        # Extract segments for this moment
        moment_start = moment.get('start', 0)
        moment_end = moment.get('end', 0)
        moment_segments = [
            seg for seg in segments
            if moment_start <= seg.get('start', 0) < moment_end
        ]
        
        if not moment_segments:
            print(f"   ⚠️  No segments found for moment")
            return None
        
        # Create clip from moment
        clip = {
            'start': moment_start,
            'end': moment_end,
            'segments': moment_segments,
            'clip_id': f"moment_{moment.get('id', 'unknown')}"
        }
        
        # Format segments for AI
        segments_text = ""
        for i, seg in enumerate(moment_segments):
            text = seg.get('text', '') or seg.get('content', '')
            segments_text += f"[{i}] {seg.get('start', 0):.1f}s-{seg.get('end', 0):.1f}s: {text[:100]}\n"
        
        # Create prompt asking for INDICES
        prompt = f"""
RESTRUCTURE CLIP - Return Segment INDICES

CURRENT CLIP ({moment_end - moment_start:.1f}s, {len(moment_segments)} segments):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{segments_text}

YOUR TASK:
Select which segments to KEEP for maximum viral potential.

RESPOND WITH JSON:
{{
  "keep_segments": [0, 2, 4, 7],  // Array of segment INDICES to keep (0-indexed)
  "reasoning": "Why I chose these segments and this order..."
}}

IMPORTANT: Return segment INDICES (numbers), not new segment objects!
Example: If you want to keep segments 0, 2, and 4, return [0, 2, 4]

Rules:
- Keep segments with strong hooks (first 3 seconds)
- Remove slow intros and filler
- Maintain story coherence
- Optimal length: 30-60 seconds
"""

        system = """You are an expert video editor optimizing clips for viral potential.

You analyze segments and select which ones to keep based on:
- Hook strength (first 3 seconds)
- Story coherence
- Pacing and flow
- Watchtime optimization

Return ONLY segment indices, not new segment objects."""

        print(f"\n   🤖 AI analyzing {len(moment_segments)} segments...")
        
        # Get AI recommendation
        result = await self.consensus.build_consensus(
            prompt=prompt,
            system=system,
            strategy='parallel_vote'
        )
        
        # Parse response
        try:
            response_text = result.get('consensus', '')
            
            # Try to extract JSON from response (multiple methods)
            # Method 1: Look for ```json block
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                if json_end > json_start:
                    json_text = response_text[json_start:json_end].strip()
                else:
                    json_text = response_text
            # Method 2: Look for first { to last }
            elif '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_text = response_text[json_start:json_end]
            else:
                # No JSON found
                print(f"   ⚠️  No JSON structure found in response")
                print(f"   Response preview: {response_text[:200]}")
                return clip
            
            # Clean common issues
            json_text = json_text.strip()
            
            # Try to parse
            try:
                restructured_data = json.loads(json_text)
            except json.JSONDecodeError as e:
                print(f"   ⚠️  JSON parsing failed: {e}")
                print(f"   Attempted JSON: {json_text[:300]}")
                
                # Try to fix common issues
                # Remove trailing commas
                json_text = json_text.replace(',]', ']').replace(',}', '}')
                
                try:
                    restructured_data = json.loads(json_text)
                    print(f"   ✅ Fixed with comma removal")
                except:
                    print(f"   ❌ Could not parse JSON, using original clip")
                    return clip
            
            # Get indices to keep (AI returns indices now, not segment objects)
            keep_indices = restructured_data.get('keep_segments', [])
            
            if not keep_indices:
                print(f"   ⚠️  AI returned no segments to keep, using original")
                return clip
            
            # Use ORIGINAL segments at those indices (preserves text field!)
            new_segments = [moment_segments[i] for i in keep_indices if i < len(moment_segments)]
            
            if not new_segments:
                print(f"   ⚠️  Invalid indices returned, using original")
                return clip
            
            print(f"   ✅ Mapped {len(keep_indices)} indices to {len(new_segments)} segments")
            
            # Verify segments have text
            for seg in new_segments:
                if 'text' not in seg and 'content' in seg:
                    seg['text'] = seg['content']
            
            # Create restructured clip with ORIGINAL segments (has text!)
            restructured = {
                **clip,
                'segments': new_segments,  # These are original segments, not AI-generated
                'start': new_segments[0]['start'],
                'end': new_segments[-1]['end'],
                'original_indices': keep_indices,
                'restructure_method': 'review',
                'ai_reasoning': restructured_data.get('reasoning', ''),
                'structure': {
                    'segments': new_segments,
                    'total_duration': new_segments[-1]['end'] - new_segments[0]['start']
                }
            }
            
            return restructured
            
        except json.JSONDecodeError as e:
            print(f"   ⚠️  JSON parsing failed: {e}")
            print(f"   Using original clip")
            return clip
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
            print(f"   Using original clip")
            return clip
    
    async def _evaluate_quality_debate(self, clip: Dict, 
                                      story_structure: Dict) -> Dict:
        """
        Quality evaluation using debate strategy WITH LEARNINGS + VALIDATION
        
        Now validates that AI actually used learnings in response
        """
        
        if not self.consensus:
            # Fallback to base system
            if self.base_system:
                return self.base_system._score_clip_quality(clip, story_structure)
            return {
                'total_score': 30,
                'quality_tier': 'C',
                'scores': {},
                'reasoning': {}
            }
        
        # GET FULL LEARNINGS (maximum quality!)
        learnings_prompt = self._get_learnings_safely(mode='full')
        
        clip_text = self._format_clip_for_eval(clip)
        
        # DEBUG: Check if clip text is valid
        if not clip_text or len(clip_text) < 50:
            print(f"   ⚠️  WARNING: Clip text is empty or too short!")
            print(f"   Length: {len(clip_text)}")
            print(f"   Content: {clip_text[:200]}")
        else:
            print(f"   ✅ Clip text formatted: {len(clip_text)} chars")
        
        prompt = f"""{learnings_prompt}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 🎬 CLIP ZU BEWERTEN:

{clip_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Bewerte (0-50) basierend auf ALLEN Learnings oben:

1. Hook Strength (0-10)
   - Nutzt einen der winning hook types?
   - Power words aus den gelernten?
   - Timing korrekt (0-3s)?

2. Story Coherence (0-10)
   - Folgt eine der winning structures?
   - Pattern interrupts alle 5-7s?
   - Loop geöffnet und geschlossen?

3. Natural Flow (0-10)
   - Keine Füller?
   - Emotionale Achterbahn?
   - Jede Sekunde gibt Grund weiterzuschauen?

4. Watchtime Potential (0-10)
   - High arousal emotions?
   - Information gap?
   - Algorithm-optimiert?

5. Emotional Impact (0-10)
   - Best emotions aus Learnings?
   - Trigger phrases?
   - Session duration impact?

Antworte in JSON mit SPEZIFISCHEN Pattern-Referenzen:
{{
  "scores": {{
    "hook_strength": X,
    "story_coherence": X,
    "natural_flow": X,
    "watchtime_potential": X,
    "emotional_impact": X
  }},
  "total_score": X,
  "quality_tier": "A/B/C/D",
  "reasoning": {{
    "strengths": [
      "Nutzt [SPEZIFISCHER PATTERN] aus Learnings",
      "Power word '[WORT]' triggert [EFFEKT]"
    ],
    "weaknesses": [
      "Missachtet [SPEZIFISCHER PATTERN]",
      "Fehlt [ELEMENT] aus Best Practices"
    ],
    "learned_patterns_applied": [
      "[PATTERN 1 NAME]",
      "[PATTERN 2 NAME]"
    ],
    "algorithm_assessment": "Watchtime/Completion/Engagement Impact erklärt"
  }}
}}
"""
        
        system = f"Du bist Qualitäts-Evaluator trainiert auf {self._get_clips_analyzed_count()} viralen Clips. Nutze die Learnings für spezifische Pattern-basierte Bewertung."
        
        # Run debate
        result = await self.consensus.build_consensus(
            prompt=prompt,
            system=system,
            strategy='debate'
        )
        
        consensus_text = result.get('consensus', '')
        
        # VALIDATE learnings were actually used
        validation = self._validate_learnings_usage(consensus_text)
        
        # Print validation report
        if validation:
            print(f"\n   🔍 Validation: {validation['confidence']:.0%} confidence")
            
            if validation['confidence'] < 0.40:
                print(f"      ⚠️  Warning: Low learnings application!")
                print(f"      Patterns: {len(validation['patterns_referenced'])}")
                print(f"      Power words: {len(validation['power_words_used'])}")
            elif validation['confidence'] >= 0.70:
                print(f"      ✅ Excellent pattern application!")
        
        # Parse JSON response
        import re
        import json
        
        json_match = re.search(r'\{[\s\S]*\}', consensus_text)
        if json_match:
            try:
                quality_result = json.loads(json_match.group())
                
                # Ensure quality_tier is set based on total_score if missing or invalid
                total_score = quality_result.get('total_score', None)
                
                # SAFE DEFAULT if parsing failed
                if total_score is None:
                    print(f"   ⚠️  Could not parse score, using confidence as fallback")
                    # Use confidence as score (0-100 → 0-50)
                    confidence = result.get('confidence', 0.5)
                    total_score = int(confidence * 50)
                    quality_result['total_score'] = total_score
                    print(f"   📊 Fallback score: {total_score}/50 (from {confidence:.0%} confidence)")
                
                if 'quality_tier' not in quality_result or not quality_result.get('quality_tier'):
                    # Safety: Handle None score
                    if total_score is None:
                        print(f"   ⚠️  Could not parse score from consensus, using confidence fallback")
                        confidence = result.get('confidence', 0.5)
                        total_score = int(confidence * 50)  # Convert confidence to score
                        print(f"   📊 Fallback score: {total_score}/50 (from {confidence:.0%} confidence)")
                        quality_result['total_score'] = total_score
                    quality_result['quality_tier'] = self._get_quality_tier(total_score)
                else:
                    # Validate tier matches score (use score-based tier if score is below threshold)
                    # Safety: Handle None score
                    if total_score is None:
                        print(f"   ⚠️  Could not parse score from consensus, using confidence fallback")
                        confidence = result.get('confidence', 0.5)
                        total_score = int(confidence * 50)  # Convert confidence to score
                        print(f"   📊 Fallback score: {total_score}/50 (from {confidence:.0%} confidence)")
                        quality_result['total_score'] = total_score
                    
                    tier_from_score = self._get_quality_tier(total_score)
                    tier_from_ai = quality_result.get('quality_tier', 'D')
                    tier_letter_ai = self._extract_tier_letter(tier_from_ai)
                    
                    # If AI tier is higher than score allows, use score-based tier
                    tier_order = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
                    if tier_order.get(tier_letter_ai, 0) > tier_order.get(tier_from_score, 0):
                        quality_result['quality_tier'] = tier_from_score
                
                quality_result['consensus_confidence'] = result.get('confidence', 0)
                quality_result['validation'] = validation
                quality_result['strategy_used'] = 'debate'
                
                # Print detailed reasoning output
                tier_string = quality_result.get('quality_tier', 'D')
                total_score = quality_result.get('total_score', 0)
                passed = tier_string in ['A', 'B', 'C']
                
                print(f"      {'✅' if passed else '❌'} {tier_string} (Score: {total_score}/50) - {'Passed' if passed else 'Rejected'}")
                
                # Print detailed reasoning from consensus
                if consensus_text:
                    print(f"\n   📝 REASONING:")
                    # Try to extract score and reasoning
                    lines = consensus_text.split('\n')
                    printed_lines = 0
                    for line in lines[:15]:  # First 15 lines
                        if line.strip() and not line.startswith('{') and not line.startswith('}'):
                            print(f"      {line[:100]}")
                            printed_lines += 1
                            if printed_lines >= 10:  # Limit to 10 lines
                                break
                    
                    if len(consensus_text) > 500:
                        print(f"\n   🔍 Full consensus (first 500 chars):")
                        print(f"      {consensus_text[:500]}...")
                
                # Also print score breakdown if available
                if 'scores' in quality_result:
                    print(f"\n   📊 Score Breakdown:")
                    scores = quality_result['scores']
                    for key, val in scores.items():
                        print(f"      {key}: {val}/10")
                
                # Print reasoning details if available
                if 'reasoning' in quality_result:
                    reasoning = quality_result['reasoning']
                    if isinstance(reasoning, dict):
                        if 'strengths' in reasoning and reasoning['strengths']:
                            print(f"\n   ✅ Strengths:")
                            for strength in reasoning['strengths'][:3]:  # Top 3
                                print(f"      • {strength[:80]}")
                        
                        if 'weaknesses' in reasoning and reasoning['weaknesses']:
                            print(f"\n   ⚠️  Weaknesses:")
                            for weakness in reasoning['weaknesses'][:3]:  # Top 3
                                print(f"      • {weakness[:80]}")
                        
                        if 'learned_patterns_applied' in reasoning and reasoning['learned_patterns_applied']:
                            print(f"\n   🎯 Patterns Applied:")
                            for pattern in reasoning['learned_patterns_applied'][:5]:  # Top 5
                                print(f"      • {pattern}")
                
                return quality_result
            except json.JSONDecodeError:
                pass
        
        # Fallback
        if self.base_system:
            return self.base_system._score_clip_quality(clip, story_structure)
        
        # Use threshold-based tier for fallback
        fallback_score = 25
        return {
            'scores': {
                'hook_strength': 5,
                'story_coherence': 5,
                'natural_flow': 5,
                'watchtime_potential': 5,
                'emotional_impact': 5
            },
            'total_score': fallback_score,
            'quality_tier': self._get_quality_tier(fallback_score),
            'reasoning': {
                'strengths': ['Could not parse evaluation'],
                'weaknesses': ['Parser error'],
                'learned_patterns_applied': [],
                'algorithm_assessment': 'N/A'
            },
            'consensus_confidence': 0.5,
            'validation': validation
        }
    
    def _get_learnings_safely(self, mode='default'):
        """
        Get learnings with graceful fallback
        
        Args:
            mode: 'default' or 'full' (full includes all details)
        """
        try:
            from master_learnings_v2 import get_learnings_for_prompt
            return get_learnings_for_prompt()
        except Exception as e:
            print(f"   ⚠️  Learnings not available: {e}")
            print(f"   ℹ️  Run: python run_learning_pipeline.py first!")
            return """
# ⚠️ LEARNINGS NOT YET TRAINED

Run initial training first:
python run_learning_pipeline.py

Then learnings will be available.
"""
    
    def _validate_learnings_usage(self, response_text: str) -> Dict:
        """Validate that AI used learnings in response"""
        try:
            from master_learnings_v2 import validate_learnings_application
            return validate_learnings_application(response_text)
        except Exception as e:
            print(f"      ⚠️  Validation failed: {e}")
            return {
                'learnings_applied': False,
                'confidence': 0.0,
                'patterns_referenced': [],
                'power_words_used': [],
                'algorithm_reasoning_present': False
            }
    
    def _print_validation_stats(self, validations: List[Dict]):
        """Print aggregated validation statistics"""
        
        if not validations:
            return
        
        print(f"\n{'='*70}")
        print("🔍 LEARNINGS VALIDATION STATS")
        print(f"{'='*70}")
        
        # Calculate averages
        confidences = [v.get('confidence', 0) for v in validations if v]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Count quality scores
        quality_scores = [v.get('quality_score', 'unknown') for v in validations if v]
        excellent = quality_scores.count('excellent')
        good = quality_scores.count('good')
        fair = quality_scores.count('fair')
        poor = quality_scores.count('poor')
        
        # Pattern usage
        all_patterns = []
        all_power_words = []
        
        for v in validations:
            if v:
                all_patterns.extend(v.get('patterns_referenced', []))
                all_power_words.extend(v.get('power_words_used', []))
        
        # Count algorithm reasoning
        algo_count = sum(1 for v in validations if v and v.get('algorithm_reasoning_present'))
        
        print(f"\n📊 OVERALL:")
        print(f"   Average Confidence: {avg_confidence:.0%}")
        print(f"   Learnings Applied: {sum(1 for v in validations if v and v.get('learnings_applied'))}/{len(validations)}")
        
        print(f"\n📈 QUALITY DISTRIBUTION:")
        print(f"   Excellent: {excellent} clips")
        print(f"   Good: {good} clips")
        print(f"   Fair: {fair} clips")
        print(f"   Poor: {poor} clips")
        
        print(f"\n🎯 PATTERN USAGE:")
        print(f"   Unique Patterns: {len(set(all_patterns))}")
        print(f"   Total References: {len(all_patterns)}")
        if all_patterns:
            from collections import Counter
            top_patterns = Counter(all_patterns).most_common(5)
            for pattern, count in top_patterns:
                print(f"      • {pattern}: {count}x")
        
        print(f"\n💪 POWER WORD USAGE:")
        print(f"   Unique Words: {len(set(all_power_words))}")
        print(f"   Total Uses: {len(all_power_words)}")
        if all_power_words:
            from collections import Counter
            top_words = Counter(all_power_words).most_common(5)
            for word, count in top_words:
                print(f"      • {word}: {count}x")
        
        print(f"\n🎯 ALGORITHM REASONING:")
        print(f"   Clips with algo reasoning: {algo_count}/{len(validations)}")
    
    def _get_clips_analyzed_count(self):
        """Get total clips analyzed from Master Learnings"""
        try:
            from master_learnings_v2 import load_master_learnings
            master = load_master_learnings()
            return master.get('metadata', {}).get('total_clips_analyzed', 972)
        except:
            return 972
    
    def _format_clip_for_eval(self, clip: Dict) -> str:
        """
        Format clip for evaluation prompt
        
        Extracts text from segments and formats for AI evaluation
        """
        
        segments = clip.get('segments', [])
        
        if not segments:
            # Fallback: try to get text directly
            fallback = clip.get('text', '') or clip.get('content', '') or ''
            if fallback:
                return f"CLIP TEXT:\n{fallback}"
            else:
                return "ERROR: No transcript available"
        
        # Extract text from segments with timestamps
        clip_text_parts = []
        
        for seg in segments:
            # Handle different segment formats
            text = seg.get('text', '') or seg.get('content', '')
            
            if text:
                start = seg.get('start', 0)
                # Format with timestamp for context
                clip_text_parts.append(f"[{start:.1f}s] {text.strip()}")
        
        # Join all parts
        if clip_text_parts:
            full_text = '\n'.join(clip_text_parts)
        else:
            # Last resort: try raw text extraction
            full_text = ' '.join([
                s.get('text', '') or s.get('content', '') 
                for s in segments 
                if s.get('text') or s.get('content')
            ])
        
        # If STILL empty, return error
        if not full_text or len(full_text.strip()) < 10:
            return f"ERROR: Could not extract text from {len(segments)} segments"
        
        # Format for evaluation
        duration = clip.get('end', 0) - clip.get('start', 0)
        
        formatted = f"""CLIP TRANSCRIPT:
Duration: {duration:.1f} seconds
Segments: {len(segments)}
Start: {clip.get('start', 0):.1f}s
End: {clip.get('end', 0):.1f}s

TRANSCRIPT:
{full_text}
"""
        
        return formatted
    
    async def _transcribe_with_assemblyai(self, video_path: Path) -> List[Dict]:
        """Try AssemblyAI transcription (simplified, matches official docs)"""
        
        try:
            import assemblyai as aai
        except ImportError:
            return None
        
        api_key = os.getenv('ASSEMBLYAI_API_KEY')
        if not api_key:
            return None
        
        aai.settings.api_key = api_key
        
        # Extract audio
        print(f"   🎵 Extracting audio from video...")
        audio_path = await self._extract_audio(video_path)
        
        if not audio_path:
            return None
        
        print(f"   ✅ Audio extracted: {audio_path.name}")
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        print(f"   📦 Audio file size: {file_size_mb:.1f} MB")
        
        print(f"   📤 Uploading to AssemblyAI...")
        print(f"   🎙️  Transcribing (this may take 3-7 minutes)...")
        
        try:
            # SIMPLE CONFIG (matches AssemblyAI official docs)
            config = aai.TranscriptionConfig(
                language_code="de"  # German only, no speaker labels
            )
            
            # Create transcriber and transcribe (synchronous)
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(str(audio_path), config=config)
            
            # Check for errors
            if transcript.status == aai.TranscriptStatus.error:
                print(f"   ❌ Transcription error: {transcript.error}")
                audio_path.unlink()
                return None
            
            print(f"   ✅ Transcription complete!")
            
            # Convert to segments using words
            segments = []
            
            if transcript.words and len(transcript.words) > 0:
                print(f"   🔧 Creating segments from {len(transcript.words)} words...")
                
                current_segment = None
                
                for word in transcript.words:
                    if not current_segment:
                        current_segment = {
                            'start': word.start / 1000.0,
                            'end': word.end / 1000.0,
                            'text': word.text,
                            'confidence': getattr(word, 'confidence', 1.0)
                        }
                    else:
                        time_gap = (word.start / 1000.0) - current_segment['end']
                        segment_duration = current_segment['end'] - current_segment['start']
                        
                        # New segment if gap > 1s or segment > 15s
                        if time_gap > 1.0 or segment_duration > 15:
                            segments.append(current_segment)
                            current_segment = {
                                'start': word.start / 1000.0,
                                'end': word.end / 1000.0,
                                'text': word.text,
                                'confidence': getattr(word, 'confidence', 1.0)
                            }
                        else:
                            current_segment['end'] = word.end / 1000.0
                            current_segment['text'] += ' ' + word.text
                
                if current_segment:
                    segments.append(current_segment)
            
            else:
                # Fallback: use transcript.text and split by sentences
                print(f"   ⚠️  No words available, using text splitting...")
                text = transcript.text
                sentences = text.split('. ')
                duration = transcript.audio_duration / 1000.0 if hasattr(transcript, 'audio_duration') else 1800
                time_per_sentence = duration / len(sentences) if sentences else 10
                
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        segments.append({
                            'start': i * time_per_sentence,
                            'end': (i + 1) * time_per_sentence,
                            'text': sentence.strip(),
                            'confidence': 1.0
                        })
            
            print(f"   ✅ Created {len(segments)} segments")
            
            # Cleanup audio file
            try:
                audio_path.unlink()
                print(f"   🗑️  Cleaned up audio file")
            except:
                pass
            
            return segments
        
        except Exception as e:
            print(f"   ❌ AssemblyAI error: {str(e)}")
            try:
                if audio_path and audio_path.exists():
                    audio_path.unlink()
            except:
                pass
            return None
    
    async def _extract_audio(self, video_path: Path) -> Optional[Path]:
        """
        Extract audio from video using moviepy
        
        Returns MP3 file (compressed) for faster AssemblyAI upload
        """
        
        try:
            # Try moviepy 2.x import first
            try:
                from moviepy import VideoFileClip
            except ImportError:
                # Fallback to moviepy 1.x
                from moviepy.editor import VideoFileClip
        except ImportError:
            print("   ❌ moviepy not installed!")
            print("   💡 Install: pip install moviepy")
            return None
        
        # Output path (MP3 for smaller file size)
        audio_path = video_path.parent / f"{video_path.stem}_audio.mp3"
        
        try:
            # Load video
            print(f"   📹 Loading video...")
            video = VideoFileClip(str(video_path))
            
            # Check if video has audio
            if not video.audio:
                print(f"   ❌ Video has no audio track!")
                video.close()
                return None
            
            # Extract and save audio (compressed for faster upload)
            print(f"   🎵 Extracting audio track (MP3, compressed)...")
            video.audio.write_audiofile(
                str(audio_path),
                fps=16000,          # 16kHz sampling rate
                bitrate='64k',      # Low bitrate (speech quality sufficient)
                codec='libmp3lame', # MP3 codec
                logger=None         # Suppress output (no verbose in moviepy 2.x)
            )
            
            # Close video
            video.close()
            
            return audio_path
        
        except Exception as e:
            print(f"   ❌ Audio extraction error: {str(e)[:100]}")
            return None
    
    async def run(self, video_path: str = None) -> Dict:
        """
        Complete integrated pipeline WITH learnings + validation
        
        Args:
            video_path: Optional direct path to video file
        """
        
        print("\n" + "="*70)
        print("🚀 CREATE CLIPS V4 - INTEGRATED PIPELINE")
        print("="*70)
        
        # Show learning stats
        try:
            clips_analyzed = self._get_clips_analyzed_count()
            print(f"\n🧠 Learning Stats:")
            print(f"   Clips analyzed: {clips_analyzed}")
            print(f"   Learnings: {'✅ Available' if get_learnings_for_prompt else '⚠️  Not trained'}")
        except:
            pass
        
        print("\n" + "="*70)
        print("📹 SELECT VIDEO")
        print("="*70)
        
        # VIDEO SELECTION - Allow direct path OR selection
        if video_path:
            # Direct path provided
            selected_video = Path(video_path)
            if not selected_video.exists():
                print(f"   ❌ Video not found: {video_path}")
                return None
            print(f"   ✅ Using: {selected_video.name}")
        
        else:
            # Show available videos
            video_dir = Path("data/uploads")
            videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.mov"))
            
            if not videos:
                print("   ❌ No videos found in data/uploads")
                print("   💡 Please add MP4/MOV files to data/uploads/")
                return None
            
            print(f"\nAvailable videos in {video_dir}:")
            for i, v in enumerate(videos, 1):
                # Check for transcript
                transcript_file = self.transcript_dir / f"{v.stem}_transcript.json"
                status = "✅ has transcript" if transcript_file.exists() else "⚠️ needs transcript"
                print(f"   {i}. {v.name} ({status})")
            
            # Also allow custom path
            print(f"   OR enter full path to video file")
            
            try:
                choice = input(f"\nSelect video (1-{len(videos)} or path): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n⚠️  Cancelled (non-interactive mode)")
                return None
            
            # Check if it's a number or path
            if choice.isdigit() and 1 <= int(choice) <= len(videos):
                selected_video = videos[int(choice) - 1]
            else:
                # Treat as path
                selected_video = Path(choice)
                if not selected_video.exists():
                    print(f"   ❌ Video not found: {choice}")
                    return None
            
            print(f"\n   ✅ Selected: {selected_video.name}")
        
        # Load or create transcript
        print("\n" + "="*70)
        print("📝 TRANSCRIPT")
        print("="*70)
        
        transcript_file = self.transcript_dir / f"{selected_video.stem}_transcript.json"
        
        if transcript_file.exists():
            print(f"   ✅ Loading cached transcript...")
            with open(transcript_file) as f:
                transcript_data = json.load(f)
            segments = transcript_data.get('segments', [])
            print(f"   ✅ Loaded {len(segments)} segments")
        
        else:
            print(f"   ⚠️  No cached transcript found")
            print(f"   🎙️  Creating transcript with AssemblyAI...")
            
            # Use AssemblyAI for transcription
            segments = await self._transcribe_with_assemblyai(selected_video)
            
            if not segments:
                print(f"   ❌ Transcription failed!")
                return None
            
            # Save transcript
            transcript_data = {
                'video_path': str(selected_video),
                'segments': segments,
                'created_at': datetime.now().isoformat(),
                'service': 'assemblyai'
            }
            
            with open(transcript_file, 'w') as f:
                json.dump(transcript_data, f, indent=2)
            
            print(f"   ✅ Transcript saved: {transcript_file}")
            print(f"   ✅ {len(segments)} segments")
        
        # Continue with pipeline...
        print("\n" + "="*70)
        print("🚀 RUNNING INTEGRATED PIPELINE")
        print("="*70)
        
        # Run integrated pipeline
        result = await self.extract_clips(segments, str(selected_video))
        
        if not result:
            print("\n❌ Pipeline failed!")
            return None
        
        # Show results
        print("\n" + "="*70)
        print("📊 RESULTS")
        print("="*70)
        print(f"\n   Story Analyzed: ✅")
        print(f"   Moments Found: {result['stats']['moments_found']}")
        print(f"   Clips Restructured: {result['stats']['clips_restructured']}")
        print(f"   Quality Passed: {result['stats']['quality_passed']}")
        print(f"   Total Versions: {result['stats']['total_versions']}")
        
        # Show quality distribution
        quality_tiers = {}
        for clip_data in result['clips']:
            tier = clip_data['clip'].get('quality', {}).get('quality_tier', '?')
            quality_tiers[tier] = quality_tiers.get(tier, 0) + 1
        
        print(f"\n   Quality Distribution:")
        for tier in ['A', 'B', 'C', 'D']:
            count = quality_tiers.get(tier, 0)
            if count > 0:
                print(f"      {tier}: {count} clips")
        
        # Export prompt
        try:
            export = input("\n📦 Export clips? (y/n): ").strip().lower()
            
            if export == 'y':
                print("\n🎬 Exporting clips...")
                
                output_dir = self.export_clips(result, str(selected_video))
                
                if output_dir:
                    print(f"\n✅ Clips exported to: {output_dir}")
                else:
                    print("\n⚠️  Export failed")
        except (EOFError, KeyboardInterrupt):
            print("\n⚠️  Skipping export (non-interactive mode)")
        
        return result
    
    def export_clips(self, result: Dict, video_path: str = None) -> str:
        """
        Export clips from V4 result format
        
        Transforms V4 format to V2 format for export
        """
        
        if not self.base_system:
            print("⚠️  Base system not available for export")
            return None
        
        # V4 format: result['clips'] = [{'clip': {...}, 'versions': [...]}]
        # V2 expects: extraction_result dict with 'clips' key and video_path as second arg
        clips_data = result.get('clips', [])
        
        if not clips_data:
            print("   ⚠️  No clips to export")
            return None
        
        # Get video_path from result or use provided
        video_path_to_use = video_path or result.get('video_path')
        
        if not video_path_to_use:
            print("   ⚠️  No video path provided")
            return None
        
        # Wrap clips_data in dict format expected by V2
        extraction_result = {
            'clips': clips_data
        }
        
        # Call base system export with both extraction_result and video_path
        output_dir = self.base_system.export_clips(
            extraction_result,
            video_path_to_use
        )
        
        return output_dir


# Main function for testing
async def test_integrated_pipeline():
    """Test the integrated pipeline with real video"""
    
    print("\n🧪 TESTING INTEGRATED PIPELINE WITH REAL VIDEO\n")
    
    # Initialize
    system = CreateClipsV4Integrated()
    
    # Get video from base system test
    video_path = "data/uploads/test.mp4"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        print("\nUsage:")
        print("  python create_clips_v4_integrated.py")
        print("  Then select video when prompted")
        return
    
    # Load transcript
    print(f"📁 Loading video: {video_path}")
    
    # Get cached transcript
    if not system.base_system:
        print("❌ Base system not available")
        return
    
    # Check for cached transcript
    cache_dir = Path("data/cache/transcripts")
    cache_files = list(cache_dir.glob("*.json")) if cache_dir.exists() else []
    
    if not cache_files:
        print("❌ No cached transcript found")
        print("\nFirst run: python create_clips_v2.py to create transcript")
        return
    
    print(f"\n📦 Found {len(cache_files)} cached transcript(s)")
    print("   Using most recent...")
    
    # Load most recent
    latest = sorted(cache_files, key=lambda p: p.stat().st_mtime)[-1]
    with open(latest) as f:
        cached = json.load(f)
        segments = cached.get('segments', [])
    
    print(f"   ✅ Loaded {len(segments)} segments")
    
    # Run integrated pipeline
    print("\n" + "="*70)
    print("🚀 RUNNING INTEGRATED PIPELINE")
    print("="*70)
    
    result = await system.extract_clips(segments, str(video_path))
    
    # Show results
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    print(f"\n   Story Analyzed: ✅")
    print(f"   Moments Found: {result['stats']['moments_found']}")
    print(f"   Clips Restructured: {result['stats']['clips_restructured']}")
    print(f"   Quality Passed: {result['stats']['quality_passed']}")
    print(f"   Total Versions: {result['stats']['total_versions']}")
    
    # Show quality distribution
    quality_tiers = {}
    for clip_data in result['clips']:
        tier = clip_data['clip'].get('quality', {}).get('quality_tier', '?')
        quality_tiers[tier] = quality_tiers.get(tier, 0) + 1
    
    print(f"\n   Quality Distribution:")
    for tier in ['A', 'B', 'C', 'D']:
        count = quality_tiers.get(tier, 0)
        if count > 0:
            print(f"      {tier}: {count} clips")
    
    # Export prompt
    try:
        export = input("\n📦 Export clips? (y/n): ").strip().lower()
        
        if export == 'y':
            print("\n🎬 Exporting clips...")
            
            # Use base system export (pass result dict, not separate args)
            output_dir = system.export_clips(result, video_path)
            
            if output_dir:
                print(f"\n✅ Clips exported to: {output_dir}")
            else:
                print("\n⚠️  Export failed")
    except (EOFError, KeyboardInterrupt):
        print("\n⚠️  Skipping export (non-interactive mode)")
    
    print("\n✅ Pipeline test complete!\n")


if __name__ == "__main__":
    import sys
    
    # Allow direct video path as argument
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    async def main():
        system = CreateClipsV4Integrated()
        result = await system.run(video_path=video_path)
        return result
    
    asyncio.run(main())

