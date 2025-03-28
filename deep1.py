import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm
import json
import requests
import warnings
import itertools

# Suppress deprecation warnings from transformers
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("conflict_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Hugging Face Inference API setup
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "YOUR_HF_API_TOKEN")  # Replace with your token
HF_API_URL = "https://api-inference.huggingface.co/models/t5-base"

def huggingface_inference(prompt, api_token=HF_API_TOKEN, api_url=HF_API_URL):
    """Call Hugging Face Inference API for free LLM inference"""
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {"inputs": prompt, "parameters": {"max_length": 256, "num_beams": 5}}
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result[0]["generated_text"] if isinstance(result, list) else "Inference failed"
    except Exception as e:
        logger.error(f"Hugging Face API error: {e}")
        return "Inference failed due to API error."

# Dataset Class
class RequirementConflictDataset(Dataset):
    """Dataset for requirement conflict detection"""
    def __init__(self, dataframe, tokenizer, max_input_length=128, max_target_length=64):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        input_text = f"Requirement: {row['Requirement_1']} | {row['Requirement_2']}"
        target_text = row["Conflict_Type"]
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": targets.input_ids.squeeze()
        }

# Generate all possible requirement pairs using n(n-1)/2
import itertools

def generate_requirement_pairs(requirements_file):
    """Generate all possible pairs from a single-column requirements file"""
    # Load requirements from file
    df = pd.read_csv(requirements_file, encoding='utf-8')
    
    # Ensure the file has a 'Requirement' column
    if "Requirement ID" not in df.columns or "Requirement Text" not in df.columns:
        logger.error("Input file must contain 'Requirement ID' and 'Requirement Text' columns.")
        return pd.DataFrame()
    
    # Format requirements as "Rxxx: Text"
    requirements = [f"{row['Requirement ID']}: {row['Requirement Text']}" for _, row in df.iterrows()]
    
    # Generate all unique pairs: n(n-1)/2
    pairs = list(itertools.combinations(requirements, 2))
    logger.info(f"Generated {len(pairs)} pairs from {len(requirements)} requirements")
    
    # Create DataFrame with pairs
    pairs_df = pd.DataFrame(pairs, columns=["Requirement_1", "Requirement_2"])
    return pairs_df

# Training Function (unchanged for now, will be used if labeled data provided)
def train_model(args):
    """Train the conflict detection model"""
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    if os.path.exists(args.input_file):
        df_train = pd.read_csv(args.input_file)
    else:
        logger.warning(f"Input file {args.input_file} not found. Generating pairs.")
        df_train = generate_requirement_pairs(args.requirements_file)
        # Note: For training, we need labeled conflict types, so this would need augmentation
    
    if args.augment:
        logger.warning("Augmentation not implemented for generated pairs yet.")
    
    train_df, val_df = train_test_split(df_train, test_size=args.validation_split, random_state=42)
    logger.info(f"Training on {len(train_df)} examples, validating on {len(val_df)} examples")
    
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)
    
    train_dataset = RequirementConflictDataset(train_df, tokenizer)
    val_dataset = RequirementConflictDataset(val_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    training_history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        training_history["train_loss"].append(avg_train_loss)
        
        model.eval()
        val_loss = 0
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                progress_bar.set_postfix({"loss": outputs.loss.item()})
        
        avg_val_loss = val_loss / len(val_loader)
        training_history["val_loss"].append(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            logger.info(f"Validation loss decreased from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model...")
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
                json.dump(training_history, f)
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    return model, tokenizer

# Prediction Function (Modified for Pairs)
def predict_conflicts(args, model=None, tokenizer=None):
    """Predict conflicts with structured output, supporting both pairwise and single-column requirement lists"""
    device = torch.device(args.device)
    use_hf_api = args.use_hf_api
    
    if not use_hf_api and (model is None or tokenizer is None):
        try:
            logger.info(f"Loading local model from {args.output_dir}")
            model = T5ForConditionalGeneration.from_pretrained(args.output_dir).to(device)
            tokenizer = T5Tokenizer.from_pretrained(args.output_dir)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return
    
    # Load requirements file
    if os.path.exists(args.test_file):
        df_input = pd.read_csv(args.test_file, encoding='utf-8')
    else:
        logger.error("Test file not found.")
        return
    
    # Check for format type (single-column or pairwise)
    if "Requirement_1" in df_input.columns and "Requirement_2" in df_input.columns:
        logger.info("Detected pairwise requirements format.")
    elif "Requirements" in df_input.columns:
        logger.info("Detected single-column requirements format. Generating requirement pairs...")
        df_input = generate_requirement_pairs(args.test_file)
    else:
        logger.error("Invalid test file format. Ensure it has either 'Requirement' or 'Requirement_1' & 'Requirement_2'.")
        return

    logger.info(f"Loaded {len(df_input)} requirement pairs")

    PREDEFINED_CONFLICTS = {
        "Performance Conflict", "Compliance Conflict", "Safety Conflict",
        "Cost Conflict", "Battery Conflict", "Environmental Conflict",
        "Structural Conflict", "Comfort Conflict", "Power Source Conflict",
        "Other"
    }

    results = []
    
    for _, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Analyzing conflicts"):
        req1 = row["Requirement_1"]
        req2 = row["Requirement_2"]
        
        input_text = (
            f"Analyze requirements conflict: {req1} AND {req2} "
            "Generate output format: Conflict_Type||Conflict_Reason||Resolution_Suggestion:"
        )
        
        if use_hf_api:
            full_output = huggingface_inference(input_text)
        else:
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=256,
                padding="max_length",
                truncation=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=256,
                    num_beams=5,
                    early_stopping=True
                )
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        parts = full_output.split("||")
        conflict_type = parts[0].strip() if len(parts) > 0 else "Other"
        if conflict_type not in PREDEFINED_CONFLICTS:
            conflict_type = "Other"
        conflict_reason = parts[1].strip() if len(parts) > 1 else "Needs manual analysis"
        resolution = parts[2].strip() if len(parts) > 2 else "Requires engineering review"
        
        results.append({
            "Requirement_1": req1,
            "Requirement_2": req2,
            "Conflict_Type": conflict_type,
            "Conflict_Reason": conflict_reason,
            "Resolution_Suggestion": resolution
        })
    
    output_df = pd.DataFrame(results)
    output_df.to_csv(args.output_file, index=False)
    logger.info(f"Analysis complete. Results saved to {args.output_file}")
    return output_df

# Main Function
def main():
    parser = argparse.ArgumentParser(description="Requirements Conflict Detection with Hugging Face API")
    
    parser.add_argument("--mode", type=str, choices=["train", "predict", "both"], default="predict")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--input_file", type=str, default="TwoWheeler_Requirement_Conflicts.csv")
    parser.add_argument("--requirements_file", type=str, default="requirements.csv", help="File with raw requirements")
    parser.add_argument("--output_dir", type=str, default="./trained_model")
    parser.add_argument("--model_name", type=str, default="t5-small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--validation_split", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--augment_multiplier", type=int, default=2)
    parser.add_argument("--test_file", type=str, default="./test_data1.csv")
    parser.add_argument("--output_file", type=str, default="conflict_results.csv")
    parser.add_argument("--use_hf_api", action="store_true", help="Use Hugging Face Inference API for predictions")
    
    args = parser.parse_args()
    
    if args.mode in ["train", "both"]:
        model, tokenizer = train_model(args)
        logger.info("Model training completed!")
    
    if args.mode in ["predict", "both"]:
        if args.mode == "both":
            predict_conflicts(args, model, tokenizer)
        else:
            predict_conflicts(args)
        logger.info("Prediction completed!")

if __name__ == "__main__":
    main()