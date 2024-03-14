import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as snspppppppppp

actual_labels = np.array(["Dog", "Not dog", "Dog", "Dog", "Not dog", "Not dog", "Not dog", "Dog", "Dog", "Not dog"])

predicted_labels = np.array(["Dog", "Not dog", "Dog", "Not dog", "Not dog", "Dog", "Not dog", "Dog", "Not dog", "Not dog"])

cm = confusion_matrix(actual_labels, predicted_labels, labels=["Dog", "Not dog"])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=["Dog", "Not dog"], yticklabels=["Dog", "Not dog"])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
