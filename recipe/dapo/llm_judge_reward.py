"""
LLM-as-a-Judge Reward Model Implementation for VERL

This module implements a reward model that uses an LLM to generate natural language
evaluation and extract scores from the generated text.
"""

import re
import json
import torch
from typing import Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMJudgeReward:
    """
    LLM-as-a-Judge reward model that generates natural language evaluation
    and extracts numerical scores.
    """
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B-Instruct",
        judge_prompt_template: str = None,
        score_extraction_pattern: str = r"Score:\s*(\d+(?:\.\d+)?)",
        use_vllm: bool = True,
        max_eval_length: int = 512,
        temperature: float = 0.1,
        device: str = "cuda",
        **kwargs
    ):
        """
        Args:
            model_path: Path to the judge LLM model
            judge_prompt_template: Template for constructing judge prompts
            score_extraction_pattern: Regex pattern to extract score from generated text
            use_vllm: Whether to use vLLM for faster inference
            max_eval_length: Maximum length of evaluation text
            temperature: Sampling temperature for generation
            device: Device to run the model on
        """
        self.model_path = model_path
        self.use_vllm = use_vllm
        self.max_eval_length = max_eval_length
        self.temperature = temperature
        self.device = device
        
        # Default judge prompt template
        if judge_prompt_template is None:
            self.judge_prompt_template = self._get_default_prompt_template()
        else:
            self.judge_prompt_template = judge_prompt_template
            
        self.score_pattern = re.compile(score_extraction_pattern)
        
        # Initialize the judge model
        self._initialize_judge_model(**kwargs)
        
    def _get_default_prompt_template(self) -> str:
        """Get default prompt template for LLM judge"""
        return """You are an AI assistant tasked with evaluating the quality of responses. Please evaluate the following response based on helpfulness, accuracy, and clarity.

User Query: {prompt}

Response to Evaluate: {response}

Please provide a detailed evaluation and assign a score from 0 to 10, where:
- 0-3: Poor quality (incorrect, unhelpful, or unclear)
- 4-6: Average quality (partially correct or helpful)
- 7-8: Good quality (mostly correct and helpful)
- 9-10: Excellent quality (accurate, helpful, and clear)

Provide your evaluation in the following format:
Evaluation: [Your detailed evaluation here]
Score: [Your numerical score]"""

    def _initialize_judge_model(self, **kwargs):
        """Initialize the judge LLM model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            if self.use_vllm:
                logger.info(f"Initializing vLLM model: {self.model_path}")
                self.model = LLM(
                    model=self.model_path,
                    tensor_parallel_size=kwargs.get('tensor_parallel_size', 1),
                    gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.8),
                    trust_remote_code=True,
                    dtype="bfloat16",
                    **kwargs
                )
                self.sampling_params = SamplingParams(
                    temperature=self.temperature,
                    max_tokens=self.max_eval_length,
                    top_p=0.9,
                )
            else:
                logger.info(f"Initializing HuggingFace model: {self.model_path}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    **kwargs
                )
                
        except Exception as e:
            logger.error(f"Failed to initialize judge model: {e}")
            raise
    
    def _format_judge_prompt(self, prompt: str, response: str) -> str:
        """Format the judge prompt with user query and response"""
        return self.judge_prompt_template.format(
            prompt=prompt.strip(),
            response=response.strip()
        )
    
    def _generate_evaluation(self, judge_prompts: list) -> list:
        """Generate evaluation text using the judge model"""
        if self.use_vllm:
            return self._generate_with_vllm(judge_prompts)
        else:
            return self._generate_with_hf(judge_prompts)
    
    def _generate_with_vllm(self, judge_prompts: list) -> list:
        """Generate evaluation using vLLM"""
        # Format prompts for chat template
        formatted_prompts = []
        for prompt in judge_prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)
        
        outputs = self.model.generate(formatted_prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def _generate_with_hf(self, judge_prompts: list) -> list:
        """Generate evaluation using HuggingFace"""
        evaluations = []
        
        for prompt in judge_prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_eval_length,
                    temperature=self.temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode only the generated part
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            evaluations.append(generated_text)
        
        return evaluations
    
    def _extract_score(self, evaluation_text: str) -> float:
        """Extract numerical score from evaluation text"""
        # Try to find score using regex pattern
        match = self.score_pattern.search(evaluation_text)
        if match:
            try:
                score = float(match.group(1))
                # Normalize to 0-1 range if needed
                if score > 1.0:
                    score = score / 10.0
                return max(0.0, min(1.0, score))
            except ValueError:
                logger.warning(f"Failed to parse score from: {match.group(1)}")
        
        # Fallback: try to find any number in the text
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', evaluation_text)
        if numbers:
            try:
                # Take the last number found (often the score)
                score = float(numbers[-1])
                if score > 1.0:
                    score = score / 10.0
                return max(0.0, min(1.0, score))
            except ValueError:
                pass
        
        # Default fallback score
        logger.warning(f"Could not extract score from evaluation: {evaluation_text[:100]}...")
        return 0.5  # Neutral score
    
    def compute_scores(
        self, 
        prompts: list, 
        responses: list, 
        return_evaluations: bool = False
    ) -> Union[list, tuple]:
        """
        Compute reward scores for a batch of prompt-response pairs
        
        Args:
            prompts: List of user prompts
            responses: List of model responses to evaluate
            return_evaluations: Whether to return the generated evaluations
            
        Returns:
            scores: List of numerical scores (0-1 range)
            evaluations: List of evaluation texts (if return_evaluations=True)
        """
        if len(prompts) != len(responses):
            raise ValueError("Number of prompts and responses must match")
        
        # Format judge prompts
        judge_prompts = [
            self._format_judge_prompt(prompt, response)
            for prompt, response in zip(prompts, responses)
        ]
        
        # Generate evaluations
        evaluations = self._generate_evaluation(judge_prompts)
        
        # Extract scores
        scores = [self._extract_score(eval_text) for eval_text in evaluations]
        
        logger.info(f"Processed {len(scores)} evaluations, scores: {scores}")
        
        if return_evaluations:
            return scores, evaluations
        return scores


def compute_score(
    data_source: str,
    solution_str: str, 
    ground_truth: str = None,
    extra_info: dict = None,
    judge_model_path: str = "Qwen/Qwen2.5-7B-Instruct",
    **kwargs
) -> Dict[str, Any]:
    """
    VERL-compatible compute_score function for LLM-as-a-Judge reward
    
    This function follows the VERL reward function interface and can be used
    as a custom reward function.
    
    Args:
        data_source: Dataset source identifier
        solution_str: The model's response to evaluate
        ground_truth: Ground truth answer (optional for LLM judge)
        extra_info: Additional information including the original prompt
        judge_model_path: Path to the judge LLM model
        
    Returns:
        Dictionary containing score and evaluation details
    """
    # Initialize judge (you might want to make this a global variable for efficiency)
    if not hasattr(compute_score, '_judge_model'):
        logger.info("Initializing LLM Judge for reward computation")
        compute_score._judge_model = LLMJudgeReward(
            model_path=judge_model_path,
            **kwargs
        )
    
    judge = compute_score._judge_model
    
    # Extract prompt from extra_info or use a default
    if extra_info and 'prompt' in extra_info:
        prompt = extra_info['prompt']
    elif extra_info and 'question' in extra_info:
        prompt = extra_info['question']
    else:
        prompt = "Please evaluate the following response."
    
    # Compute score
    scores, evaluations = judge.compute_scores(
        prompts=[prompt],
        responses=[solution_str],
        return_evaluations=True
    )
    
    score = scores[0]
    evaluation = evaluations[0]
    
    return {
        'score': score,
        'evaluation': evaluation,
        'judge_model': judge_model_path,
        'data_source': data_source
    }


# Example usage for standalone testing
if __name__ == "__main__":
    # Example usage
    judge = LLMJudgeReward(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        use_vllm=True
    )
    
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms."
    ]
    
    responses = [
        "The capital of France is Paris.",
        "Quantum computing is a very complex topic that I cannot explain."
    ]
    
    scores, evaluations = judge.compute_scores(prompts, responses, return_evaluations=True)
    
    for i, (prompt, response, score, evaluation) in enumerate(zip(prompts, responses, scores, evaluations)):
        print(f"\n=== Example {i+1} ===")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print(f"Score: {score}")
        print(f"Evaluation: {evaluation}") 