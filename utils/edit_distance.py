"""
Edit distance utilities for text correction.

Provides Levenshtein and Damerau-Levenshtein distance calculations
for non-destructive text correction.
"""

from typing import List, Tuple


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Levenshtein distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Damerau-Levenshtein distance (includes transpositions).
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Damerau-Levenshtein distance
    """
    if len(s1) < len(s2):
        return damerau_levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    # Create distance matrix
    d = {}
    maxdist = len(s1) + len(s2)
    d[-1, -1] = maxdist
    
    for i in range(len(s1) + 1):
        d[i, -1] = maxdist
        d[i, 0] = i
    for j in range(len(s2) + 1):
        d[-1, j] = maxdist
        d[0, j] = j
    
    # Dictionary of last occurrence of each character
    last_row = {}
    
    for i in range(len(s1)):
        last_match_col = 0
        for j in range(len(s2)):
            last_match_row = last_row.get(s2[j], 0)
            cost = 1 if s1[i] != s2[j] else 0
            d[i, j] = min(
                d[i - 1, j] + 1,  # deletion
                d[i, j - 1] + 1,  # insertion
                d[i - 1, j - 1] + cost  # substitution
            )
            if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                # Transposition
                d[i, j] = min(d[i, j], d[last_match_row - 1, last_match_col - 1] + (i - last_match_row - 1) + 1 + (j - last_match_col - 1))
            last_match_col = j
        last_row[s1[i]] = i
    
    return d[len(s1) - 1, len(s2) - 1]


def find_closest_match(word: str, candidates: List[str], max_distance: int = 3) -> Tuple[Optional[str], int]:
    """
    Find closest match for a word from candidate list.
    
    Args:
        word: Word to match
        candidates: List of candidate words
        max_distance: Maximum allowed edit distance
        
    Returns:
        Tuple of (closest_match, distance) or (None, max_distance) if no match found
    """
    best_match = None
    best_distance = max_distance + 1
    
    for candidate in candidates:
        distance = damerau_levenshtein_distance(word.lower(), candidate.lower())
        if distance < best_distance:
            best_distance = distance
            best_match = candidate
            if distance == 0:
                break
    
    if best_distance <= max_distance:
        return best_match, best_distance
    else:
        return None, best_distance
