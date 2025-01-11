class QABenchmark:
    """
    A class to calculate and evaluate the accuracy of LLM-generated answers
    against standard reference answers.
    """
    
    def __init__(self):
        self.scoring_weights = {
            'content_inclusion': 2,
            'contextual_alignment': 2,
            'clarity': 2
        }
    
    def calculate_content_inclusion(self, standard_answer, llm_answer):
        """
        Calculate content inclusion score based on overlap of key points.
        
        Args:
            standard_answer (str): Reference answer
            llm_answer (str): Generated answer
            
        Returns:
            float: Score between 0-2
        """
        # Basic implementation - count overlapping words
        standard_words = set(standard_answer.lower().split())
        llm_words = set(llm_answer.lower().split())
        overlap = len(standard_words.intersection(llm_words))
        total = len(standard_words)
        
        # Scale to 0-2 range
        if total == 0:
            return 0
        return min(2, (overlap / total) * 2)
    
    def calculate_contextual_alignment(self, standard_answer, llm_answer):
        """
        Evaluate how well the LLM answer maintains the context and meaning
        of the standard answer using NLP techniques.
        
        Args:
            standard_answer (str): Reference answer
            llm_answer (str): Generated answer
            
        Returns:
            float: Score between 0-2
        """
        import nltk
        from nltk.corpus import wordnet as wn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        # Download all required NLTK resources
        required_resources = [
            'punkt',
            'wordnet',
            'averaged_perceptron_tagger',
            'punkt_tab',
            'averaged_perceptron_tagger_eng'
        ]
        
        for resource in required_resources:
            try:
                nltk.download(resource, quiet=True)
            except:
                print(f"Warning: Failed to download NLTK resource: {resource}")
        
        def get_tfidf_similarity(text1, text2):
            """Calculate cosine similarity using TF-IDF vectors"""
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([text1, text2])
            return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        
        def get_semantic_similarity(text1, text2):
            """Calculate semantic similarity using WordNet"""
            tokens1 = nltk.word_tokenize(text1)
            tokens2 = nltk.word_tokenize(text2)
            
            pos_map = {'N': wn.NOUN, 'V': wn.VERB, 'R': wn.ADV, 'J': wn.ADJ}
            
            def get_synsets(word, pos_tag):
                pos = pos_map.get(pos_tag[0], None)
                return wn.synsets(word, pos=pos) if pos else []
            
            # Get POS tags
            pos1 = nltk.pos_tag(tokens1)
            pos2 = nltk.pos_tag(tokens2)
            
            # Calculate similarity for matching words
            similarities = []
            for (word1, tag1), (word2, tag2) in zip(pos1, pos2):
                synsets1 = get_synsets(word1, tag1)
                synsets2 = get_synsets(word2, tag2)
                
                if synsets1 and synsets2:
                    max_sim = max(
                        (s1.path_similarity(s2) or 0)
                        for s1 in synsets1
                        for s2 in synsets2
                    )
                    similarities.append(max_sim)
            
            return np.mean(similarities) if similarities else 0
        
        # Calculate scores
        tfidf_score = get_tfidf_similarity(standard_answer, llm_answer)
        semantic_score = get_semantic_similarity(standard_answer, llm_answer)
        
        # Combine scores with weights
        combined_score = (0.6 * tfidf_score) + (0.4 * semantic_score)
        
        # Scale to 0-2 range
        return min(2.0, max(0.0, combined_score * 2))
    
    def calculate_clarity(self, llm_answer):
        """
        Evaluate the clarity and coherence of the LLM answer using
        state-of-the-art readability metrics and linguistic analysis.
        
        Args:
            llm_answer (str): Generated answer
            
        Returns:
            float: Score between 0-2
        """
        import textstat
        import numpy as np
        from collections import Counter
        
        # Calculate readability scores
        flesch = textstat.flesch_reading_ease(llm_answer)
        ari = textstat.automated_readability_index(llm_answer)
        
        # Analyze sentence structure
        sentences = textstat.sentence_count(llm_answer)
        words = textstat.lexicon_count(llm_answer, removepunct=True)
        avg_sentence_length = words / sentences if sentences > 0 else 0
        
        # Calculate word complexity
        syllables = textstat.syllable_count(llm_answer)
        avg_syllables_per_word = syllables / words if words > 0 else 0
        
        # Calculate lexical diversity
        unique_words = len(set(llm_answer.lower().split()))
        lexical_diversity = unique_words / words if words > 0 else 0
        
        # Normalize scores to 0-1 range
        def normalize(score, min_val, max_val):
            return min(1, max(0, (score - min_val) / (max_val - min_val)))
        
        # Readability score (0-100 scale)
        readability = normalize(flesch, 0, 100)
        
        # Sentence complexity (inverse of average sentence length)
        sentence_complexity = 1 - normalize(avg_sentence_length, 5, 30)
        
        # Word complexity (inverse of syllables per word)
        word_complexity = 1 - normalize(avg_syllables_per_word, 1, 3)
        
        # Combine metrics with weights
        clarity_score = (
            0.4 * readability + 
            0.3 * sentence_complexity + 
            0.2 * word_complexity + 
            0.1 * lexical_diversity
        )
        
        # Scale to 0-2 range
        return min(2.0, max(0.0, clarity_score * 2))
    
    def calculate_accuracy(self, standard_answer, llm_answer):
        """
        Calculate overall accuracy score based on weighted components.
        
        Args:
            standard_answer (str): Reference answer
            llm_answer (str): Generated answer
            
        Returns:
            dict: Dictionary containing scores and final accuracy
        """
        content_score = self.calculate_content_inclusion(standard_answer, llm_answer)
        context_score = self.calculate_contextual_alignment(standard_answer, llm_answer)
        clarity_score = self.calculate_clarity(llm_answer)
        
        total_weight = sum(self.scoring_weights.values())
        accuracy = (content_score + context_score + clarity_score) / total_weight
        
        return {
            'content_inclusion': content_score,
            'contextual_alignment': context_score,
            'clarity': clarity_score,
            'accuracy': accuracy
        }

def process_csv(input_file, output_file):
    """
    Process a CSV file containing standard answers and multiple LLM answers.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to save results CSV file
    """
    import pandas as pd
    import os
    from datetime import datetime
    
    benchmark = QABenchmark()
    
    # Set base directory
    base_dir = r"C:\\Users\\limti\\Desktop\\qa-benchmark"
    
    # Create full paths
    input_path = os.path.join(base_dir, input_file)
    output_path = os.path.join(base_dir, output_file)
    
    # Check if input file exists
    if not os.path.exists(input_path):
        # Create template CSV if it doesn't exist
        template = pd.DataFrame(columns=[
            'standard_answer',
            'llm1_answer',
            'llm2_answer'
        ])
        template.to_csv(input_path, index=False)
        print(f"Created template input file at: {input_path}")
        print("Please populate the CSV with your data and run again.")
        return
    
    # Read input CSV
    try:
        df = pd.read_csv(input_path, encoding='utf-8')
    except Exception as e:
        print(f"Error reading input file: {e}")
        return
    
    # Prepare results storage
    results = []
    
    # Process each row
    for index, row in df.iterrows():
        standard_answer = row['standard_answer']
        llm1_answer = row['llm1_answer']
        llm2_answer = row['llm2_answer']
        
        # Calculate scores for both LLMs
        llm1_results = benchmark.calculate_accuracy(standard_answer, llm1_answer)
        llm2_results = benchmark.calculate_accuracy(standard_answer, llm2_answer)
        
        # Store results
        results.append({
            'question_id': index + 1,
            'llm1_content_inclusion': llm1_results['content_inclusion'],
            'llm1_contextual_alignment': llm1_results['contextual_alignment'],
            'llm1_clarity': llm1_results['clarity'],
            'llm1_accuracy': llm1_results['accuracy'],
            'llm2_content_inclusion': llm2_results['content_inclusion'],
            'llm2_contextual_alignment': llm2_results['contextual_alignment'],
            'llm2_clarity': llm2_results['clarity'],
            'llm2_accuracy': llm2_results['accuracy']
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate averages
    averages = results_df.mean(numeric_only=True)
    averages['question_id'] = 'Average'
    results_df = pd.concat([results_df, pd.DataFrame([averages])], ignore_index=True)
    
    # Ensure output directory exists using full path
    output_path = os.path.join(base_dir, output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Print results to terminal
    print("\n=== Benchmark Results ===")
    print(results_df.to_string(index=False))
    print("\n=== Averages ===")
    print(results_df.iloc[-1:].to_string(index=False))
    
    # Save results
    try:
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")
    except Exception as e:
        print(f"\nError saving results: {e}")

def main():
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process Q&A accuracy benchmark')
    parser.add_argument('input_file', help='Name of input CSV file in qa-benchmark directory')
    parser.add_argument('output_file', help='Name of output CSV file in qa-benchmark directory')
    
    args = parser.parse_args()
    
    # Process the CSV file
    process_csv(args.input_file, args.output_file)

if __name__ == "__main__":
    main()