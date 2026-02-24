"""
Prompting utilities for multiple-choice QA.
Improved with true batching, Few-Shot, and Chain-of-Thought (CoT) support.
"""
import torch
import reTransformerForMultipleChoice
from torch import Tensor
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part3.nn_utils import softmax

class PromptTemplate:
    TEMPLATES = {
        "basic": "Context: {context}\n\nQuestion: {question}\n\nChoices:\n{choices_formatted}\n\nAnswer:",
        "expert": "You are an expert reading comprehension solver. Read the passage and select the correct letter for the question.\n\nPassage: {context}\n\nQuestion: {question}\n\nChoices:\n{choices_formatted}\n\nCorrect Answer:",
        "cot": "Context: {context}\n\nQuestion: {question}\n\nChoices:\n{choices_formatted}\n\nLet's think step by step to find the correct answer.\n\nReasoning:"
    }
    
    def __init__(self, template_name: str = "basic", custom_template: Optional[str] = None, choice_format: str = "letter"):
        self.template_name = template_name
        self.template = custom_template if custom_template else self.TEMPLATES.get(template_name, self.TEMPLATES["basic"])
        self.choice_format = choice_format
    
    def _format_choices(self, choices: List[str]) -> str:
        labels = ["A", "B", "C", "D", "E", "F", "G", "H"] if self.choice_format == "letter" else [str(i+1) for i in range(len(choices))]
        return "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
    
    def format(self, context: str, question: str, choices: List[str], few_shot_examples: Optional[List[Dict]] = None, **kwargs) -> str:
        prompt = ""
        
        # 1. Prepend few-shot examples if provided
        if few_shot_examples:
            for ex in few_shot_examples:
                prompt += self.template.format(
                    context=ex["context"], 
                    question=ex["question"], 
                    choices_formatted=self._format_choices(ex["choices"]), 
                    **kwargs
                )
                
                # If using CoT, the few-shot examples should ideally contain a "reasoning" key
                if self.template_name == "cot" and "reasoning" in ex:
                    prompt += f" {ex['reasoning']}\n"
                    
                # Append the final correct answer
                answer_label = chr(ord('A') + ex["answer"]) if self.choice_format == "letter" else str(ex["answer"] + 1)
                
                if self.template_name == "cot":
                    prompt += f"Therefore, the correct choice is: {answer_label}\n\n"
                else:
                    prompt += f" {answer_label}\n\n"

        # 2. Format the actual target question
        prompt += self.template.format(
            context=context, 
            question=question, 
            choices_formatted=self._format_choices(choices), 
            **kwargs
        )
        return prompt


class PromptingPipeline:
    def __init__(self, model, tokenizer, template: Optional[PromptTemplate] = None, device: str = "cuda"):
        self.model = model.to(device) if hasattr(model, 'to') else model
        self.tokenizer = tokenizer
        self.template = template or PromptTemplate("basic")
        self.device = device

        # MY REVISION: Dynamically grab max length from your model
        if hasattr(model, 'config'):
            self.max_length = model.config.get('context_length', 512) if isinstance(model.config, dict) else getattr(model.config, 'context_length', 512)
        else:
            self.max_length = 512
        
        # We need a pad token for true batching (fallback to EOS if pad doesn't exist)
        self.pad_token_id = getattr(self.tokenizer, 'pad_token_id', None) or getattr(self.tokenizer, 'eos_token_id', 0)
        self._setup_choice_tokens()
    
    def _setup_choice_tokens(self):
        """Extract exact token IDs for choices A, B, C, D to calculate probabilities."""
        self.choice_tokens = {}
        for label in ["A", "B", "C", "D", "E", "F"]:
            for prefix in ["", " ", "\n"]:
                try:
                    # Try to encode safely depending on tokenizer API
                    token_ids = self.tokenizer.encode(prefix + label, add_special_tokens=False)
                except TypeError:
                    token_ids = self.tokenizer.encode(prefix + label)
                    
                if token_ids:
                    self.choice_tokens[label] = token_ids[-1]
                    break

    def _build_safe_prompts(self, batch_ex: List[Dict], few_shot_examples: Optional[List[Dict]], max_generated_tokens: int = 0) -> List[str]:
        """Dynamically truncates ONLY the context to fit within the model's max length."""
        prompts = []
        for ex in batch_ex:
            # 1. Format a "dummy" prompt to measure the few-shot, question, and choices
            dummy_prompt = self.template.format(context="", question=ex["question"], choices=ex["choices"], few_shot_examples=few_shot_examples)
            dummy_tokens = self.tokenizer.encode(dummy_prompt)
            
            # 2. Calculate remaining space (minus a safety buffer of 5 for special tokenizer tokens)
            max_ctx_len = self.max_length - len(dummy_tokens) - max_generated_tokens - 5
            
            # 3. Encode the context, and truncate ONLY the context if it's too long
            ctx_tokens = self.tokenizer.encode(ex["context"])
            if len(ctx_tokens) > max_ctx_len and max_ctx_len > 0:
                ctx_tokens = ctx_tokens[:max_ctx_len] 
                
            # 4. Decode the safe context back to string and build the final prompt
            safe_context = self.tokenizer.decode(ctx_tokens)
            safe_prompt = self.template.format(context=safe_context, question=ex["question"], choices=ex["choices"], few_shot_examples=few_shot_examples)
            prompts.append(safe_prompt)
            
        return prompts

    @torch.no_grad()
    def predict_batch(self, examples: List[Dict[str, Any]], batch_size: int = 8, few_shot_examples: Optional[List[Dict]] = None) -> List[int]:
        """Fast batched inference looking only at the next-token logits (Standard / Few-shot)."""
        self.model.eval()
        all_predictions = []
        
        for i in range(0, len(examples), batch_size):
            batch_ex = examples[i:i + batch_size]
            
            # Build prompts safely (truncating context only)
            prompts = self._build_safe_prompts(batch_ex, few_shot_examples, max_generated_tokens=0)
            
            # Left-pad sequences so the final token is aligned for logit extraction
            encoded = [self.tokenizer.encode(p) for p in prompts]
            max_len = max(len(seq) for seq in encoded)
            padded_input_ids = [([self.pad_token_id] * (max_len - len(seq))) + seq for seq in encoded]
            
            input_tensor = torch.tensor(padded_input_ids, device=self.device)
            logits = self.model(input_tensor)[:, -1, :] # Grab the logits for the very last token
            
            for b_idx, ex in enumerate(batch_ex):
                choice_labels = ["A", "B", "C", "D", "E", "F"][:len(ex["choices"])]
                choice_logits = [
                    logits[b_idx, self.choice_tokens[label]].item() if label in self.choice_tokens else float("-inf")
                    for label in choice_labels
                ]
                probs = softmax(torch.tensor(choice_logits), dim=-1)
                all_predictions.append(probs.argmax().item())
                
        return all_predictions

    @torch.no_grad()
    def _generate_greedy(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Manually generates tokens autoregressively.
        Assumes self.model(input_ids) returns logits of shape [batch, seq_len, vocab_size].
        """
        eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)
        
        # Track which sequences in the batch have hit the EOS token
        batch_size = input_ids.size(0)
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)
        
        for _ in range(max_new_tokens):
            # 1. Forward pass (get logits)
            logits = self.model(input_ids)
            
            # 2. Get the logits for the very last generated token
            next_token_logits = logits[:, -1, :]
            
            # 3. Greedy selection: pick the token with the highest logit
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # 4. Handle End-of-Sequence (EOS)
            if eos_token_id is not None:
                # If a sequence is already finished, just output the pad token
                next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (1 - unfinished_sequences)
                # Mark newly finished sequences
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())
            
            # 5. Append the newly generated tokens to the input sequence
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # 6. Stop early if every sequence in the batch has emitted an EOS token
            if eos_token_id is not None and unfinished_sequences.max() == 0:
                break
                
        return input_ids
    
    @torch.no_grad()
    def predict_batch_cot(self, examples: List[Dict[str, Any]], batch_size: int = 4, max_new_tokens: int = 150, few_shot_examples: Optional[List[Dict]] = None) -> List[int]:
        """Chain-of-Thought batched inference using autoregressive generation and regex parsing."""
        self.model.eval()
        all_predictions = []
        
        for i in range(0, len(examples), batch_size):
            batch_ex = examples[i:i + batch_size]
            
            # Build prompts safely, leaving room for max_new_tokens
            prompts = self._build_safe_prompts(batch_ex, few_shot_examples, max_generated_tokens=max_new_tokens)
            
            # Left-pad
            encoded = [self.tokenizer.encode(p) for p in prompts]
            max_len = max(len(seq) for seq in encoded)
            padded_input_ids = [([self.pad_token_id] * (max_len - len(seq))) + seq for seq in encoded]
            
            input_tensor = torch.tensor(padded_input_ids, device=self.device)
            attention_mask = (input_tensor != self.pad_token_id).long()
            
            # Generate the reasoning text
            output_sequences = self._generate_greedy(
                input_ids=input_tensor,
                max_new_tokens=max_new_tokens
            )
            
            for b_idx, ex in enumerate(batch_ex):
                # Isolate the newly generated tokens
                generated_tokens = output_sequences[b_idx][max_len:]
                generated_text = self.tokenizer.decode(generated_tokens.tolist())
                
                predicted_idx = self._extract_answer_from_cot(generated_text, len(ex["choices"]))
                all_predictions.append(predicted_idx)
                
        return all_predictions

    def _extract_answer_from_cot(self, text: str, num_choices: int) -> int:
        """Parses the generated reasoning to find the final selected letter."""
        patterns = [
            r"Therefore, the correct choice is:?\s*([A-F])",
            r"The correct answer is:?\s*([A-F])",
            r"Answer:\s*([A-F])",
            r"\*\*([A-F])\*\*",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                idx = ord(match.group(1).upper()) - ord('A')
                if 0 <= idx < num_choices:
                    return idx
                    
        # Fallback: just grab the very last standalone letter A-F in the text
        fallback_match = re.findall(r'\b([A-F])\b', text, re.IGNORECASE)
        if fallback_match:
            idx = ord(fallback_match[-1].upper()) - ord('A')
            if 0 <= idx < num_choices:
                return idx
                
        return 0 # Default to A if parsing totally fails

    @torch.no_grad()
    def predict_single(self, context: str, question: str, choices: List[str], use_cot: bool = False):
        examples = [{"context": context, "question": question, "choices": choices}]
        if use_cot:
            return self.predict_batch_cot(examples, batch_size=1)[0]
        return self.predict_batch(examples, batch_size=1)[0]


def evaluate_prompting(pipeline, examples: List[Dict[str, Any]], batch_size: int = 8, use_cot: bool = False, few_shot_examples: Optional[List[Dict]] = None) -> Dict[str, Any]:
    if use_cot:
        predictions = pipeline.predict_batch_cot(examples, batch_size=batch_size, few_shot_examples=few_shot_examples)
    else:
        predictions = pipeline.predict_batch(examples, batch_size=batch_size, few_shot_examples=few_shot_examples)
        
    correct = sum(1 for p, ex in zip(predictions, examples) if ex.get("answer", -1) >= 0 and p == ex["answer"])
    total = sum(1 for ex in examples if ex.get("answer", -1) >= 0)
    
    return {"accuracy": correct / total if total > 0 else 0.0, "predictions": predictions}