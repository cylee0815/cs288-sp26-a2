"""
Neural network utilities for Transformer implementation.
Contains basic building blocks: softmax, cross-entropy, gradient clipping, token accuracy, perplexity.
"""
import torch
from torch import Tensor


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Compute softmax along the specified dimension.
    
    Args:
        x: Input tensor of any shape
        dim: Dimension along which to compute softmax (default: -1)
    
    Returns:
        Tensor of same shape as input with softmax applied along dim
    """
    # TODO: Implement numerically stable softmax. You can re-use the same one 
    # used in part 2. But for this problem, you need to implement a numerically stable version to pass harder tests.
    # Subtract max for numerical stability (prevents overflow when doing exp)
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - x_max)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)
    # raise NotImplementedError("Implement softmax")


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Compute cross-entropy loss.
    
    Args:
        logits: Unnormalized log probabilities of shape (N, C) where N is batch size
                and C is number of classes
        targets: Ground truth class indices of shape (N,)
    
    Returns:
        Scalar tensor containing the mean cross-entropy loss
    """
    # TODO: Implement cross-entropy loss
    # 1. Numerically stable log_softmax: log(exp(x - max) / sum(exp(x - max)))
    # which simplifies to: (x - max) - log(sum(exp(x - max)))
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
    safe_logits = logits - max_logits
    
    log_sum_exp = torch.log(torch.sum(torch.exp(safe_logits), dim=-1, keepdim=True))
    log_probs = safe_logits - log_sum_exp
    
    # 2. Extract the log probability of the true target classes
    # logits shape: (N, C), targets shape: (N,)
    batch_size = logits.shape[0]
    batch_indices = torch.arange(batch_size, device=logits.device)
    
    # Gather the log_probs for the specific target indices
    target_log_probs = log_probs[batch_indices, targets]
    
    # 3. NLL Loss is the negative mean of the target log probabilities
    return -torch.mean(target_log_probs)
    # raise NotImplementedError("Implement cross_entropy")


def gradient_clipping(parameters, max_norm: float) -> Tensor:
    """
    Clip gradients of parameters by global norm.
    
    Args:
        parameters: Iterable of parameters with gradients
        max_norm: Maximum allowed gradient norm
    
    Returns:
        The total norm of the gradients before clipping
    """
    # TODO: Implement gradient clipping
    # Collect all parameters that have gradients
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.tensor(0.0)
    
    # Calculate the global L2 norm across all parameter gradients
    # total_norm = sqrt(sum(norm(grad)^2))
    norms = [torch.norm(p.grad.detach(), p=2.0) for p in params]
    total_norm = torch.norm(torch.stack(norms), p=2.0)
    
    # Compute scaling coefficient
    # Add small epsilon to prevent division by zero
    clip_coef = max_norm / (total_norm + 1e-6)
    
    # If the total norm exceeds the max norm, scale gradients down
    if clip_coef < 1.0:
        for p in params:
            p.grad.detach().mul_(clip_coef)
            
    return total_norm
    # raise NotImplementedError("Implement gradient_clipping")


def token_accuracy(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """
    Compute token-level accuracy for language modeling.
    
    Computes the fraction of tokens where the predicted token (argmax of logits)
    matches the target token, ignoring positions where target equals ignore_index.
    
    Args:
        logits: Predicted logits of shape (N, C) where N is the number of tokens
                and C is the vocabulary size
        targets: Ground truth token indices of shape (N,)
        ignore_index: Target value to ignore when computing accuracy (default: -100)
    
    Returns:
        Scalar tensor containing the accuracy (between 0 and 1)
    
    Example:
        >>> logits = torch.tensor([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2], [1.0, 0.5, 2.5]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> token_accuracy(logits, targets)
        tensor(1.)  # All predictions correct: argmax gives [0, 1, 2]
        
        >>> logits = torch.tensor([[2.0, 1.0], [0.1, 3.0], [1.0, 0.5]])
        >>> targets = torch.tensor([1, 1, 0])
        >>> token_accuracy(logits, targets)
        tensor(0.6667)  # 2 out of 3 correct
    """
    # TODO: Implement token accuracy
    # Find the predicted class (index with highest logit)
    preds = torch.argmax(logits, dim=-1)
    
    # Create a mask to filter out ignored indices
    valid_mask = (targets != ignore_index)
    
    # Check predictions against targets where mask is valid
    correct = (preds == targets) & valid_mask
    
    # Sum of correct predictions divided by total valid tokens
    valid_tokens = valid_mask.sum()
    
    # Prevent division by zero if all targets are ignored
    if valid_tokens == 0:
        return torch.tensor(0.0, device=logits.device)
        
    return correct.sum().float() / valid_tokens.float()
    # raise NotImplementedError("Implement token_accuracy")


def perplexity(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """
    Compute perplexity for language modeling.
    
    Perplexity is defined as exp(cross_entropy_loss). It measures how well the
    probability distribution predicted by the model matches the actual distribution
    of the tokens. Lower perplexity indicates better prediction.
    
    Args:
        logits: Predicted logits of shape (N, C) where N is the number of tokens
                and C is the vocabulary size
        targets: Ground truth token indices of shape (N,)
        ignore_index: Target value to ignore when computing perplexity (default: -100)
    
    Returns:
        Scalar tensor containing the perplexity (always >= 1)
    
    Example:
        >>> # Perfect predictions (one-hot logits matching targets)
        >>> logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> perplexity(logits, targets)
        tensor(1.0001)  # Close to 1 (perfect)
        
        >>> # Uniform predictions (high uncertainty)
        >>> logits = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> perplexity(logits, targets)
        tensor(3.)  # Equal to vocab_size (worst case for uniform)
    """
    # TODO: Implement perplexity
    # Create a mask to filter out ignored indices
    valid_mask = (targets != ignore_index)
    
    # Filter both logits and targets
    valid_logits = logits[valid_mask]
    valid_targets = targets[valid_mask]
    
    # Handle the edge case where everything is masked out
    if valid_logits.numel() == 0:
        return torch.tensor(1.0, device=logits.device)
        
    # Calculate Cross Entropy Loss on valid tokens
    loss = cross_entropy(valid_logits, valid_targets)
    
    # Perplexity = exp(Loss)
    return torch.exp(loss)
    # raise NotImplementedError("Implement perplexity")
