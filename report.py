import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

### VISUALIZATION
def report(final_model, X_train, y_train, X_test, y_test):
	predictions = final_model.predict(X_test)

	cm = confusion_matrix(y_test, predictions, labels=[0, 1])

	matrix_display = ConfusionMatrixDisplay(
		confusion_matrix=cm,
		display_labels=['Not Depressed', 'Depressed']
	)
	matrix_display.plot(cmap='Blues')
	plt.title("Logistic Regression – Confusion Matrix")
	plt.tight_layout() # fixes text clipping
	plt.show()