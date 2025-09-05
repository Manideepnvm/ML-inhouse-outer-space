import numpy as np
import pandas as pd
import warnings
import os
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow import keras
    from tensorflow import keras
    from tensorflow import layers, models, callbacks, optimizers
    from keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
    
    # Set TensorFlow logging level
    tf.get_logger().setLevel('ERROR')
    
except ImportError:
    TENSORFLOW_AVAILABLE = False

# PyTorch imports with error handling
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. PyTorch models will be skipped.")

warnings.filterwarnings('ignore')

class DeepLearningTrainer:
    """
    Deep learning model trainer for astronomical classification.
    """
    
    def __init__(self, num_classes=3):
        """
        Initialize the deep learning trainer.
        
        Args:
            num_classes (int): Number of classification classes
        """
        self.num_classes = num_classes
        self.models = {}
        self.history = {}
        self.label_encoder = None
        
    def create_dense_model(self, input_dim, hidden_layers=[128, 64, 32], dropout_rate=0.3):
        """
        Create a dense neural network model.
        
        Args:
            input_dim (int): Number of input features
            hidden_layers (list): List of hidden layer sizes
            dropout_rate (float): Dropout rate
            
        Returns:
            keras.Model: Dense neural network
        """
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available")
            return None
        
        model = models.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for i, units in enumerate(hidden_layers):
            model.add(layers.Dense(units, activation='relu', name=f'dense_{i+1}'))
            model.add(layers.BatchNormalization(name=f'batch_norm_{i+1}'))
            model.add(layers.Dropout(dropout_rate, name=f'dropout_{i+1}'))
        
        # Output layer
        if self.num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid', name='output'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax', name='output'))
            loss = 'categorical_crossentropy'
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def create_deep_model(self, input_dim, dropout_rate=0.4):
        """
        Create a deeper neural network model.
        
        Args:
            input_dim (int): Number of input features
            dropout_rate (float): Dropout rate
            
        Returns:
            keras.Model: Deep neural network
        """
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available")
            return None
        
        model = models.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        
        # First block
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        
        # Second block
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        
        # Third block
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        
        # Fourth block
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        if self.num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax'))
            loss = 'categorical_crossentropy'
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def create_wide_deep_model(self, input_dim):
        """
        Create a wide & deep model architecture.
        
        Args:
            input_dim (int): Number of input features
            
        Returns:
            keras.Model: Wide & Deep model
        """
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available")
            return None
        
        # Input layer
        inputs = layers.Input(shape=(input_dim,))
        
        # Wide component (linear)
        wide = layers.Dense(self.num_classes, activation='linear', name='wide')(inputs)
        
        # Deep component
        deep = layers.Dense(128, activation='relu')(inputs)
        deep = layers.BatchNormalization()(deep)
        deep = layers.Dropout(0.3)(deep)
        
        deep = layers.Dense(64, activation='relu')(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.Dropout(0.3)(deep)
        
        deep = layers.Dense(32, activation='relu')(deep)
        deep = layers.Dense(self.num_classes, activation='linear', name='deep')(deep)
        
        # Combine wide and deep
        combined = layers.Add()([wide, deep])
        
        if self.num_classes == 2:
            outputs = layers.Activation('sigmoid')(combined)
            loss = 'binary_crossentropy'
        else:
            outputs = layers.Activation('softmax')(combined)
            loss = 'categorical_crossentropy'
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def create_autoencoder_classifier(self, input_dim, encoding_dim=32):
        """
        Create an autoencoder-based classifier.
        
        Args:
            input_dim (int): Number of input features
            encoding_dim (int): Encoding dimension
            
        Returns:
            keras.Model: Autoencoder classifier
        """
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available")
            return None
        
        # Input layer
        inputs = layers.Input(shape=(input_dim,))
        
        # Encoder
        encoded = layers.Dense(128, activation='relu')(inputs)
        encoded = layers.Dense(64, activation='relu')(encoded)
        encoded = layers.Dense(encoding_dim, activation='relu', name='encoding')(encoded)
        
        # Decoder
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dense(128, activation='relu')(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # Classifier head
        classifier = layers.Dense(64, activation='relu')(encoded)
        classifier = layers.Dropout(0.3)(classifier)
        
        if self.num_classes == 2:
            classifier = layers.Dense(1, activation='sigmoid')(classifier)
            loss = {'output_class': 'binary_crossentropy', 'output_recon': 'mse'}
        else:
            classifier = layers.Dense(self.num_classes, activation='softmax')(classifier)
            loss = {'output_class': 'categorical_crossentropy', 'output_recon': 'mse'}
        
        # Create model with two outputs
        model = models.Model(
            inputs=inputs, 
            outputs={'output_class': classifier, 'output_recon': decoded}
        )
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=loss,
            loss_weights={'output_class': 1.0, 'output_recon': 0.1},
            metrics={'output_class': 'accuracy'}
        )
        
        return model
    
    class PyTorchMLP(nn.Module):
        """PyTorch Multi-Layer Perceptron."""
        
        def __init__(self, input_dim, num_classes, hidden_dims=[128, 64, 32]):
            super().__init__()
            self.layers = nn.ModuleList()
            
            # Input layer
            prev_dim = input_dim
            
            # Hidden layers
            for hidden_dim in hidden_dims:
                self.layers.append(nn.Linear(prev_dim, hidden_dim))
                self.layers.append(nn.BatchNorm1d(hidden_dim))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(0.3))
                prev_dim = hidden_dim
            
            # Output layer
            self.layers.append(nn.Linear(prev_dim, num_classes))
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    def train_pytorch_model(self, X_train, y_train, X_val=None, y_val=None, 
                           epochs=50, batch_size=32, learning_rate=0.001):
        """
        Train a PyTorch model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation target
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            
        Returns:
            object: Trained PyTorch model
        """
        if not PYTORCH_AVAILABLE:
            print("❌ PyTorch not available")
            return None
        
        print("🔥 Training PyTorch MLP...")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Create model
        model = self.PyTorchMLP(X_train.shape[1], self.num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        train_losses = []
        train_accuracies = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            epoch_acc = 100 * correct / total
            train_losses.append(epoch_loss / len(train_loader))
            train_accuracies.append(epoch_acc)
            
            # Validation if provided
            if X_val is not None and y_val is not None:
                model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(X_val)
                    y_val_tensor = torch.LongTensor(y_val)
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_acc = 100 * (val_predicted == y_val_tensor).sum().item() / y_val_tensor.size(0)
                
                scheduler.step(val_loss)
                
                if epoch % 10 == 0:
                    print(f"   Epoch {epoch}: Train Loss: {train_losses[-1]:.4f}, "
                          f"Train Acc: {epoch_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            else:
                if epoch % 10 == 0:
                    print(f"   Epoch {epoch}: Train Loss: {train_losses[-1]:.4f}, Train Acc: {epoch_acc:.2f}%")
        
        # Store training history
        self.history['pytorch_mlp'] = {
            'loss': train_losses,
            'accuracy': train_accuracies
        }
        
        return model
    
    def prepare_data_for_keras(self, X, y):
        """
        Prepare data for Keras training.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Target
            
        Returns:
            tuple: Prepared (X, y) for Keras
        """
        X_prep = X.astype(np.float32)
        
        if self.num_classes == 2:
            y_prep = y.astype(np.float32)
        else:
            y_prep = to_categorical(y, num_classes=self.num_classes)
        
        return X_prep, y_prep
    
    def train_keras_model(self, model_name, model, X_train, y_train, 
                         X_val=None, y_val=None, epochs=50, batch_size=32):
        """
        Train a Keras model.
        
        Args:
            model_name (str): Name of the model
            model (keras.Model): Keras model to train
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation target
            epochs (int): Number of epochs
            batch_size (int): Batch size
            
        Returns:
            keras.Model: Trained model
        """
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available")
            return None
        
        print(f"🧠 Training {model_name}...")
        
        # Prepare data
        X_train_prep, y_train_prep = self.prepare_data_for_keras(X_train, y_train)
        
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_prep, y_val_prep = self.prepare_data_for_keras(X_val, y_val)
            validation_data = (X_val_prep, y_val_prep)
        
        # Callbacks
        callback_list = [
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5, patience=5, min_lr=1e-7, verbose=0
            ),
            callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10, restore_best_weights=True, verbose=0
            )
        ]
        
        # Train model
        try:
            history = model.fit(
                X_train_prep, y_train_prep,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                callbacks=callback_list,
                verbose=0
            )
            
            # Store history
            self.history[model_name] = history.history
            
            print(f"   ✅ {model_name} training completed!")
            if validation_data:
                final_val_acc = history.history['val_accuracy'][-1]
                print(f"   📊 Final validation accuracy: {final_val_acc:.4f}")
            
            return model
            
        except Exception as e:
            print(f"   ❌ Error training {model_name}: {e}")
            return None
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None, 
                        epochs=50, batch_size=32):
        """
        Train all deep learning models.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation target
            epochs (int): Number of epochs
            batch_size (int): Batch size
            
        Returns:
            dict: Dictionary of trained models
        """
        print("\n🧠 TRAINING DEEP LEARNING MODELS")
        print("=" * 60)
        
        input_dim = X_train.shape[1]
        
        # Create validation split if not provided
        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            X_train = X_train_split
            y_train = y_train_split
            print("   📊 Created validation split (80/20)")
        
        # Dense Neural Network
        if TENSORFLOW_AVAILABLE:
            dense_model = self.create_dense_model(input_dim)
            if dense_model is not None:
                trained_dense = self.train_keras_model(
                    'dense_nn', dense_model, X_train, y_train, 
                    X_val, y_val, epochs, batch_size
                )
                if trained_dense is not None:
                    self.models['dense_nn'] = trained_dense
            
            # Deep Neural Network
            deep_model = self.create_deep_model(input_dim)
            if deep_model is not None:
                trained_deep = self.train_keras_model(
                    'deep_nn', deep_model, X_train, y_train, 
                    X_val, y_val, epochs, batch_size
                )
                if trained_deep is not None:
                    self.models['deep_nn'] = trained_deep
            
            # Wide & Deep Model
            wide_deep_model = self.create_wide_deep_model(input_dim)
            if wide_deep_model is not None:
                trained_wd = self.train_keras_model(
                    'wide_deep', wide_deep_model, X_train, y_train, 
                    X_val, y_val, epochs, batch_size
                )
                if trained_wd is not None:
                    self.models['wide_deep'] = trained_wd
            
            # Autoencoder Classifier
            autoencoder_model = self.create_autoencoder_classifier(input_dim)
            if autoencoder_model is not None:
                # Special training for autoencoder (needs reconstruction target)
                X_train_prep, y_train_prep = self.prepare_data_for_keras(X_train, y_train)
                X_val_prep, y_val_prep = self.prepare_data_for_keras(X_val, y_val)
                
                try:
                    history = autoencoder_model.fit(
                        X_train_prep, 
                        {'output_class': y_train_prep, 'output_recon': X_train_prep},
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(
                            X_val_prep, 
                            {'output_class': y_val_prep, 'output_recon': X_val_prep}
                        ),
                        callbacks=[
                            callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=0),
                            callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=0)
                        ],
                        verbose=0
                    )
                    
                    self.models['autoencoder'] = autoencoder_model
                    self.history['autoencoder'] = history.history
                    print("   ✅ Autoencoder classifier training completed!")
                    
                except Exception as e:
                    print(f"   ❌ Error training autoencoder: {e}")
        
        # PyTorch MLP
        if PYTORCH_AVAILABLE:
            pytorch_model = self.train_pytorch_model(
                X_train, y_train, X_val, y_val, epochs, batch_size
            )
            if pytorch_model is not None:
                self.models['pytorch_mlp'] = pytorch_model
        
        print(f"\n🎉 Deep learning training completed! {len(self.models)} models trained.")
        return self.models
    
    def predict(self, model_name, X_test):
        """
        Make predictions using a trained model.
        
        Args:
            model_name (str): Name of the model
            X_test (np.ndarray): Test features
            
        Returns:
            np.ndarray: Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        if model_name == 'pytorch_mlp' and PYTORCH_AVAILABLE:
            # PyTorch prediction
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test)
                outputs = model(X_tensor)
                _, predictions = torch.max(outputs, 1)
                return predictions.numpy()
        
        elif TENSORFLOW_AVAILABLE and hasattr(model, 'predict'):
            # Keras prediction
            X_test_prep = X_test.astype(np.float32)
            
            if model_name == 'autoencoder':
                # Special handling for autoencoder
                predictions = model.predict(X_test_prep, verbose=0)
                if isinstance(predictions, dict):
                    pred_proba = predictions['output_class']
                else:
                    pred_proba = predictions[0]  # First output is classification
            else:
                pred_proba = model.predict(X_test_prep, verbose=0)
            
            if self.num_classes == 2:
                return (pred_proba > 0.5).astype(int).flatten()
            else:
                return np.argmax(pred_proba, axis=1)
        
        else:
            raise ValueError(f"Cannot make predictions with model '{model_name}'")
    
    def predict_proba(self, model_name, X_test):
        """
        Get prediction probabilities.
        
        Args:
            model_name (str): Name of the model
            X_test (np.ndarray): Test features
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        if model_name == 'pytorch_mlp' and PYTORCH_AVAILABLE:
            # PyTorch prediction probabilities
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test)
                outputs = model(X_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                return probabilities.numpy()
        
        elif TENSORFLOW_AVAILABLE and hasattr(model, 'predict'):
            # Keras prediction probabilities
            X_test_prep = X_test.astype(np.float32)
            
            if model_name == 'autoencoder':
                predictions = model.predict(X_test_prep, verbose=0)
                if isinstance(predictions, dict):
                    return predictions['output_class']
                else:
                    return predictions[0]  # First output is classification
            else:
                return model.predict(X_test_prep, verbose=0)
        
        else:
            raise ValueError(f"Cannot get probabilities from model '{model_name}'")
    
    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a trained model.
        
        Args:
            model_name (str): Name of the model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
            
        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.predict(model_name, X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall'],
            'f1_macro': report['macro avg']['f1-score'],
            'precision_weighted': report['weighted avg']['precision'],
            'recall_weighted': report['weighted avg']['recall'],
            'f1_weighted': report['weighted avg']['f1-score']
        }
        
        return metrics
    
    def save_models(self, save_dir='models/'):
        """
        Save all trained models.
        
        Args:
            save_dir (str): Directory to save models
        """
        print(f"\n💾 SAVING DEEP LEARNING MODELS TO {save_dir}")
        print("-" * 40)
        
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'pytorch_mlp' and PYTORCH_AVAILABLE:
                    # Save PyTorch model
                    model_path = os.path.join(save_dir, f'{model_name}_model.pth')
                    torch.save(model.state_dict(), model_path)
                    
                    # Save model architecture info
                    arch_info = {
                        'input_dim': model.layers[0].in_features,
                        'num_classes': self.num_classes,
                        'hidden_dims': [layer.out_features for layer in model.layers 
                                       if isinstance(layer, nn.Linear)][:-1]
                    }
                    arch_path = os.path.join(save_dir, f'{model_name}_architecture.joblib')
                    import joblib
                    joblib.dump(arch_info, arch_path)
                    
                elif TENSORFLOW_AVAILABLE and hasattr(model, 'save'):
                    # Save Keras model
                    model_path = os.path.join(save_dir, f'{model_name}_model.h5')
                    model.save(model_path)
                
                print(f"   ✅ {model_name} saved")
                
            except Exception as e:
                print(f"   ❌ Error saving {model_name}: {e}")
        
        # Save training history
        if self.history:
            history_path = os.path.join(save_dir, 'dl_training_history.joblib')
            import joblib
            # Convert numpy arrays to lists for serialization
            history_serializable = {}
            for model_name, hist in self.history.items():
                history_serializable[model_name] = {}
                for key, value in hist.items():
                    if isinstance(value, np.ndarray):
                        history_serializable[model_name][key] = value.tolist()
                    elif isinstance(value, list):
                        history_serializable[model_name][key] = value
                    else:
                        history_serializable[model_name][key] = [value]
            
            joblib.dump(history_serializable, history_path)
            print(f"   ✅ Training history saved")
    
    def load_models(self, load_dir='models/'):
        """
        Load saved models.
        
        Args:
            load_dir (str): Directory to load models from
        """
        print(f"\n📂 LOADING DEEP LEARNING MODELS FROM {load_dir}")
        print("-" * 40)
        
        if not os.path.exists(load_dir):
            print(f"❌ Model directory not found: {load_dir}")
            return
        
        # Load Keras models
        if TENSORFLOW_AVAILABLE:
            keras_files = [f for f in os.listdir(load_dir) if f.endswith('.h5')]
            for model_file in keras_files:
                model_name = model_file.replace('_model.h5', '')
                model_path = os.path.join(load_dir, model_file)
                
                try:
                    self.models[model_name] = keras.models.load_model(model_path)
                    print(f"   ✅ {model_name} loaded")
                except Exception as e:
                    print(f"   ❌ Error loading {model_name}: {e}")
        
        # Load PyTorch models
        if PYTORCH_AVAILABLE:
            pytorch_files = [f for f in os.listdir(load_dir) if f.endswith('.pth')]
            for model_file in pytorch_files:
                model_name = model_file.replace('_model.pth', '')
                model_path = os.path.join(load_dir, model_file)
                arch_path = os.path.join(load_dir, f'{model_name}_architecture.joblib')
                
                if os.path.exists(arch_path):
                    try:
                        import joblib
                        arch_info = joblib.load(arch_path)
                        
                        # Recreate model architecture
                        model = self.PyTorchMLP(
                            arch_info['input_dim'],
                            arch_info['num_classes'],
                            arch_info['hidden_dims']
                        )
                        
                        # Load weights
                        model.load_state_dict(torch.load(model_path))
                        model.eval()
                        
                        self.models[model_name] = model
                        print(f"   ✅ {model_name} loaded")
                        
                    except Exception as e:
                        print(f"   ❌ Error loading {model_name}: {e}")
        
        # Load training history
        history_path = os.path.join(load_dir, 'dl_training_history.joblib')
        if os.path.exists(history_path):
            try:
                import joblib
                self.history = joblib.load(history_path)
                print(f"   ✅ Training history loaded")
            except Exception as e:
                print(f"   ⚠️ Error loading training history: {e}")
    
    def get_model_summary(self):
        """
        Get summary of all trained models.
        
        Returns:
            pd.DataFrame: Model summary
        """
        if not self.models:
            return pd.DataFrame()
        
        summary_data = []
        
        for model_name, model in self.models.items():
            # Get model info
            if model_name == 'pytorch_mlp' and PYTORCH_AVAILABLE:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            elif TENSORFLOW_AVAILABLE and hasattr(model, 'count_params'):
                total_params = model.count_params()
                trainable_params = total_params
            else:
                total_params = 'Unknown'
                trainable_params = 'Unknown'
            
            # Get final training metrics
            final_loss = 'N/A'
            final_acc = 'N/A'
            
            if model_name in self.history:
                hist = self.history[model_name]
                if 'loss' in hist and hist['loss']:
                    final_loss = hist['loss'][-1]
                if 'accuracy' in hist and hist['accuracy']:
                    final_acc = hist['accuracy'][-1]
                elif 'output_class_accuracy' in hist and hist['output_class_accuracy']:
                    final_acc = hist['output_class_accuracy'][-1]
            
            summary_data.append({
                'Model': model_name,
                'Framework': 'PyTorch' if model_name == 'pytorch_mlp' else 'TensorFlow',
                'Total_Parameters': total_params,
                'Trainable_Parameters': trainable_params,
                'Final_Loss': final_loss,
                'Final_Accuracy': final_acc
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_training_history(self, model_name=None):
        """
        Plot training history for models.
        
        Args:
            model_name (str): Specific model to plot (None = all models)
        """
        if not self.history:
            print("⚠️ No training history available")
            return
        
        import matplotlib.pyplot as plt
        
        models_to_plot = [model_name] if model_name else list(self.history.keys())
        
        fig, axes = plt.subplots(len(models_to_plot), 2, figsize=(15, 5*len(models_to_plot)))
        if len(models_to_plot) == 1:
            axes = axes.reshape(1, -1)
        
        for i, model_name in enumerate(models_to_plot):
            if model_name not in self.history:
                continue
                
            hist = self.history[model_name]
            
            # Plot loss
            if 'loss' in hist:
                axes[i, 0].plot(hist['loss'], label='Training Loss')
                if 'val_loss' in hist:
                    axes[i, 0].plot(hist['val_loss'], label='Validation Loss')
                axes[i, 0].set_title(f'{model_name} - Loss')
                axes[i, 0].set_xlabel('Epoch')
                axes[i, 0].set_ylabel('Loss')
                axes[i, 0].legend()
                axes[i, 0].grid(True)
            
            # Plot accuracy
            acc_key = 'accuracy'
            if acc_key not in hist and 'output_class_accuracy' in hist:
                acc_key = 'output_class_accuracy'
            
            if acc_key in hist:
                axes[i, 1].plot(hist[acc_key], label='Training Accuracy')
                val_acc_key = f'val_{acc_key}'
                if val_acc_key in hist:
                    axes[i, 1].plot(hist[val_acc_key], label='Validation Accuracy')
                axes[i, 1].set_title(f'{model_name} - Accuracy')
                axes[i, 1].set_xlabel('Epoch')
                axes[i, 1].set_ylabel('Accuracy')
                axes[i, 1].legend()
                axes[i, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results/{model_name if model_name else "all"}_training_history.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_ensemble_model(self, X_train, y_train, model_list=None):
        """
        Create an ensemble of deep learning models.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            model_list (list): List of models to ensemble
            
        Returns:
            object: Ensemble model
        """
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow required for ensemble creation")
            return None
        
        if model_list is None:
            model_list = [name for name in self.models.keys() if 'pytorch' not in name]
        
        if len(model_list) < 2:
            print("⚠️ Need at least 2 models for ensemble")
            return None
        
        print(f"🎭 Creating ensemble from models: {model_list}")
        
        # Create ensemble input
        input_dim = X_train.shape[1]
        inputs = layers.Input(shape=(input_dim,))
        
        # Get predictions from each model
        model_outputs = []
        for model_name in model_list:
            if model_name in self.models and model_name != 'autoencoder':
                # Create a copy of the model's architecture for ensemble
                model = self.models[model_name]
                # Extract features from the pre-output layer
                feature_extractor = models.Model(
                    inputs=model.input,
                    outputs=model.layers[-2].output
                )
                features = feature_extractor(inputs)
                model_outputs.append(features)
        
        if not model_outputs:
            print("❌ No compatible models found for ensemble")
            return None
        
        # Combine features
        if len(model_outputs) == 1:
            combined = model_outputs[0]
        else:
            combined = layers.Concatenate()(model_outputs)
        
        # Final classification layer
        ensemble_output = layers.Dense(64, activation='relu')(combined)
        ensemble_output = layers.Dropout(0.3)(ensemble_output)
        
        if self.num_classes == 2:
            ensemble_output = layers.Dense(1, activation='sigmoid')(ensemble_output)
            loss = 'binary_crossentropy'
        else:
            ensemble_output = layers.Dense(self.num_classes, activation='softmax')(ensemble_output)
            loss = 'categorical_crossentropy'
        
        ensemble_model = models.Model(inputs=inputs, outputs=ensemble_output)
        ensemble_model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss=loss,
            metrics=['accuracy']
        )
        
        return ensemble_model

            