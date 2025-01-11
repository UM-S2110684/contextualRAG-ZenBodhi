class QnAEvaluationMetric:
    def __init__(self):
        self.metrics = {
            'contextual_accuracy': {
                'description': 'Ability to correctly identify and incorporate tradition-specific elements',
                'max_score': 2
            },
            'depth_of_understanding': {
                'description': 'Reflection of nuances without oversimplification',
                'max_score': 2
            },
            'avoidance_of_cross_contextual_errors': {
                'description': 'Maintaining clear distinctions between traditions',
                'max_score': 2
            },
            'faithfulness_to_sources': {
                'description': 'Appropriate references to tradition-specific texts/doctrines',
                'max_score': 2
            }
        }
        self.total_max_score = sum(m['max_score'] for m in self.metrics.values())

    def evaluate_response(self, response, reference):
        """
        Evaluate a response against reference criteria
        Returns a dictionary with scores and percentage
        """
        scores = {}
        
        # Contextual Accuracy
        scores['contextual_accuracy'] = self._evaluate_contextual_accuracy(
            response, reference)
            
        # Depth of Understanding
        scores['depth_of_understanding'] = self._evaluate_depth_of_understanding(
            response, reference)
            
        # Avoidance of Cross-Contextual Errors
        scores['avoidance_of_cross_contextual_errors'] = self._evaluate_cross_contextual_errors(
            response, reference)
            
        # Faithfulness to Sources
        scores['faithfulness_to_sources'] = self._evaluate_faithfulness_to_sources(
            response, reference)
            
        # Calculate total score and percentage
        total_score = sum(scores.values())
        percentage = (total_score / self.total_max_score) * 100
        
        return {
            'scores': scores,
            'total_score': total_score,
            'percentage': percentage,
            'max_score': self.total_max_score
        }

    def _evaluate_contextual_accuracy(self, response, reference):
        """
        Evaluate if response correctly identifies and incorporates tradition-specific elements
        using keyword analysis with lightweight semantic similarity
        """
        score = 0
        
        # First try keyword-based scoring
        for tradition in reference['traditions']:
            tradition_score = 0
            expected_elements = reference['expected_elements'][tradition]
            
            # Count exact matches
            matches = sum(1 for element in expected_elements 
                         if element.lower() in response.lower())
            
            # Calculate match percentage
            match_percentage = matches / len(expected_elements)
            
            if match_percentage >= 0.4:
                tradition_score = 2
            elif match_percentage >= 0.2:
                tradition_score = 1
                
            # Enhanced semantic similarity check
            if tradition_score < 2:
                response_lower = response.lower()
                for element in expected_elements:
                    element_lower = element.lower()
                    
                    # Exact match already handled, check for partial matches
                    if element_lower not in response_lower:
                        # Split into words and check for partial matches
                        element_words = element_lower.split()
                        matched = False
                        
                        # Check each word in element
                        for word in element_words:
                            # Check for partial matches with minimum length
                            if len(word) > 3:  # Only check words longer than 3 characters
                                # Look for partial matches in response
                                for response_word in response_lower.split():
                                    if len(response_word) > 3 and word in response_word:
                                        # Calculate match ratio
                                        match_ratio = len(word) / len(response_word)
                                        if match_ratio >= 0.7:  # Require at least 70% match
                                            tradition_score = min(tradition_score + 0.5, 2)
                                            matched = True
                                            break
                                if matched:
                                    break
                            
            score = max(score, tradition_score)
                
        return int(score)

    def _evaluate_depth_of_understanding(self, response, reference):
        """
        Evaluate response depth using multiple factors:
        1. Length and structure
        2. Term density and variety
        3. Comparative analysis
        """
        score = 0
        words = response.split()
        
        # 1. Length and Structure Analysis
        if len(words) >= 200:  # Comprehensive response
            score += 1
        elif len(words) < 50:  # Too short
            return 0
            
        # Count paragraphs and sections (rough estimate)
        paragraphs = len([p for p in response.split('\n') if p.strip()])
        if paragraphs >= 4:  # Well-structured response
            score += 0.5
            
        # 2. Term Density and Variety
        unique_terms = set()
        tradition_coverage = {t: 0 for t in reference['traditions']}
        
        for tradition in reference['traditions']:
            for element in reference['expected_elements'][tradition]:
                if element.lower() in response.lower():
                    unique_terms.add(element.lower())
                    tradition_coverage[tradition] += 1
        
        # Calculate term density
        term_density = len(unique_terms) / len(words)
        if term_density >= 0.03:
            score += 0.5
            
        # Check balanced coverage of traditions
        min_coverage = min(tradition_coverage.values())
        if min_coverage >= 2:  # At least 2 terms from each tradition
            score += 0.5
            
        # 3. Check for comparative language
        comparative_terms = ['whereas', 'while', 'however', 'in contrast', 'unlike', 'different', 'similar', 'both']
        if any(term in response.lower() for term in comparative_terms):
            score += 0.5
            
        # Convert continuous score to discrete score
        if score >= 2:
            return 2
        elif score >= 1:
            return 1
        return 0

    def _evaluate_cross_contextual_errors(self, response, reference):
        """
        Detect cross-tradition confusion using keyword analysis and context
        """
        # Initialize score at maximum
        score = 2
        
        # 1. Check for clear tradition separation
        tradition_sections = {t: [] for t in reference['traditions']}
        paragraphs = [p.strip() for p in response.split('\n') if p.strip()]
        
        for p in paragraphs:
            p_lower = p.lower()
            matches = {
                t: sum(1 for term in reference['expected_elements'][t] 
                      if term.lower() in p_lower)
                for t in reference['traditions']
            }
            max_matches = max(matches.values())
            if max_matches > 0:
                matching_traditions = [t for t, m in matches.items() if m == max_matches]
                if len(matching_traditions) == 1:
                    tradition_sections[matching_traditions[0]].append(p)
        
        # Reduce score if traditions aren't clearly separated
        if not all(tradition_sections.values()):
            score -= 1
        
        # 2. Check for comparative language
        comparative_terms = ['whereas', 'while', 'however', 'in contrast', 'unlike', 'different']
        has_comparison = any(term in response.lower() for term in comparative_terms)
        
        # 3. Check for confusion indicators
        confusion_terms = ['same', 'similar', 'both traditions', 'equally']
        has_confusion = any(term in response.lower() for term in confusion_terms)
        
        # Adjust score based on comparison and confusion
        if has_comparison and not has_confusion:
            score = max(score, 1)
        elif has_confusion and not has_comparison:
            score = min(score, 1)
            
        return score

    def _evaluate_faithfulness_to_sources(self, response, reference):
        """
        Verify source references using text matching and context analysis
        """
        score = 0
        response_lower = response.lower()
        
        # 1. Check for direct source citations
        source_terms = ['sutra', 'text', 'path', 'practice', 'tradition', 'teaching']
        tradition_sources = {t: [] for t in reference['traditions']}
        
        for tradition in reference['traditions']:
            for element in reference['expected_elements'][tradition]:
                if any(term in element.lower() for term in source_terms):
                    tradition_sources[tradition].append(element)
        
        # Count source matches with context
        valid_citations = 0
        total_citations = 0
        
        for tradition, sources in tradition_sources.items():
            if not sources:
                continue
                
            total_citations += len(sources)
            for source in sources:
                source_lower = source.lower()
                if source_lower in response_lower:
                    # Check if source is mentioned in appropriate context
                    context_start = max(0, response_lower.find(source_lower) - 50)
                    context_end = min(len(response_lower), 
                                    response_lower.find(source_lower) + len(source_lower) + 50)
                    context = response_lower[context_start:context_end]
                    
                    if tradition.lower() in context:
                        valid_citations += 1
        
        # 2. Calculate score based on valid citations and total sources
        if total_citations == 0:
            return 1
            
        citation_ratio = valid_citations / total_citations
        
        if citation_ratio >= 0.4:
            score = 2
        elif citation_ratio >= 0.2:
            score = 1
            
        return score

    def compare_responses(self, response1, response2, reference):
        """
        Compare two responses and return their evaluation results
        """
        eval1 = self.evaluate_response(response1, reference)
        eval2 = self.evaluate_response(response2, reference)
        
        return {
            'response1': eval1,
            'response2': eval2,
            'comparison': {
                'score_difference': eval1['total_score'] - eval2['total_score'],
                'percentage_difference': eval1['percentage'] - eval2['percentage']
            }
        }

# Example usage
if __name__ == "__main__":
    evaluator = QnAEvaluationMetric()
    
    # Example responses demonstrating high-quality (Norbu) vs comparison (ZenBodhi) responses
    norbu_response = """
    In Theravāda Buddhism, compassion is developed through specific meditative practices outlined in the Visuddhimagga, particularly through the cultivation of the brahma-vihāras (divine abodes). The tradition emphasizes individual liberation through the arhat path, where compassion emerges naturally through deep meditation and understanding of the Four Noble Truths.

    In contrast, the Mahāyāna tradition, as exemplified in the Bodhicaryāvatāra, places compassion at the center of the bodhisattva ideal. The text emphasizes the practice of exchanging self for others and seeing all beings as interconnected. This approach is rooted in the philosophical understanding of śūnyatā (emptiness) and the bodhisattva's vow to liberate all beings.
    """

    zenbodhi_response = """
    Buddhist traditions emphasize compassion through various practices. The Visuddhimagga mentions meditation techniques, while the bodhisattva path focuses on helping others. Both Theravāda and Mahāyāna share similar goals but have different approaches to practice. The brahma-vihāras and bodhisattva practices are important methods for developing kindness and understanding.
    """
    
    # Comprehensive reference criteria for evaluation
    reference = {
        'traditions': ['Theravāda', 'Mahāyāna'],
        'expected_elements': {
            'Theravāda': [
                'brahma-vihāras',
                'Visuddhimagga',
                'arhat path',
                'individual liberation',
                'Four Noble Truths',
                'meditation practice'
            ],
            'Mahāyāna': [
                'Bodhicaryāvatāra',
                'bodhisattva ideal',
                'exchanging self for others',
                'śūnyatā',
                'emptiness',
                'interconnected beings',
                'bodhisattva vow'
            ]
        }
    }
    
    comparison = evaluator.compare_responses(norbu_response, zenbodhi_response, reference)
