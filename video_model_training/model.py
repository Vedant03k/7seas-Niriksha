import torch
import torch.nn as nn
import timm

class VideoDeepfakeModel(nn.Module):
    def __init__(self, sequence_length=15, lstm_hidden_size=512, lstm_layers=1, dropout=0.5):
        """
        Spatial-Temporal Hybrid Deepfake Detection Model
        - Spatial: Pre-trained Xception (Feature Extractor)
        - Temporal: LSTM (Captures frame-to-frame inconsistencies)
        """
        super(VideoDeepfakeModel, self).__init__()
        
        self.sequence_length = sequence_length
        
        # 1. Spatial Extractor: EfficientNet-B4 (pretrained on ImageNet)
        # num_classes=0 removes the final Dense layer so it returns the raw feature vectors
        self.feature_extractor = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=0)
        
        # Get the feature dimension automatically
        feature_dim = self.feature_extractor.num_features
        
        # Freeze Backbone initially so we don't destroy its pre-trained weights during early training
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # 2. Temporal Analyzer: LSTM
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True, # Input format: (batch, seq, feature)
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        
        # 3. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
            # Removed nn.Sigmoid() here to output raw logits (required for stable Mixed Precision Training)
        )

    def forward(self, x):
        """
        Input: Tensor of shape (Batch_Size, Sequence_Length, Channels(3), Height(299), Width(299))
        Output: Tensor of shape (Batch_Size, 1) -> FAKE Probability
        """
        batch_size, seq_length, c, h, w = x.size()
        
        # Reshape to flatten batch and sequence for CNN processing
        # Output shape: (Batch * Seq_Len, C, H, W)
        x = x.view(batch_size * seq_length, c, h, w)
        
        # 1. Feature Extraction (Xception)
        # Extract features for all frames across all batches
        features = self.feature_extractor(x)
        
        # Reshape back to sequence format for LSTM
        # Output shape: (Batch, Seq_Len, feature_dim)
        features = features.view(batch_size, seq_length, -1)
        
        # 2. Temporal Analysis (LSTM)
        # lstm_out shape: (Batch, Seq_Len, lstm_hidden)
        lstm_out, (hidden_state, cell_state) = self.lstm(features)
        
        # We only care about the final prediction after the entire sequence is processed
        last_hidden_output = lstm_out[:, -1, :] # Shape: (Batch, lstm_hidden)
        
        # 3. Final Classification
        out = self.classifier(last_hidden_output) # Shape: (Batch, 1)
        
        return out

    def unfreeze_backbone(self, num_layers=5):
        """
        Call this to unfreeze the top layers of the backbone during fine-tuning (Stage 2).
        """
        for count, child in enumerate(reversed(list(self.feature_extractor.children()))):
            if count < num_layers:
                for param in child.parameters():
                    param.requires_grad = True

if __name__ == "__main__":
    # Test to ensure the model compiles and shapes match perfectly with dummy data
    print("Testing Model Compilation & Forward Pass...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VideoDeepfakeModel().to(device)
    
    # Dummy Input mimicking our Video Data Tensor: 2 videos, 15 frames each, 3 channel (RGB), 299x299 resolution
    print("Creating dummy tensor shape: (Batch=2, Seq=15, C=3, H=299, W=299)")
    dummy_input = torch.randn(2, 15, 3, 299, 299).to(device)
    
    # Run a forward pass
    output = model(dummy_input)
    print(f"Model Forward successful! Final Output Shape: {output.shape} (Outputs: {output.detach().cpu().numpy().flatten()})")
    print(f"To represent FAKE probabilities for the 2 videos.")
