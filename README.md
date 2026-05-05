# Deep_Learning_Assignment
This project uses an LSTM Autoencoder to detect mechanical failures before they happen. By training on "healthy" sensor sequences, the model learns to reconstruct normal vibration patterns. If a fault occurs, the model fails to accurately recreate the signal; this high reconstruction loss serves as a real-time trigger for maintenance.
