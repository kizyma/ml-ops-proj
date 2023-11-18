import model_card_toolkit as mct
from datetime import date
from model_card_toolkit import ModelCardToolkit, Owner, Reference, Risk, Limitation, UseCase, User
import uuid

# Initialize the Model Card Toolkit with a path to store generated assets
output_dir = "./experiments/model_cards/model_card_assets"
toolkit = ModelCardToolkit(output_dir=output_dir)

# Create a model card
model_card = toolkit.scaffold_assets()

# Basic Information
model_card.model_details.name = "Toxicity Classification Model"
model_card.model_details.overview = (
    "This model predicts the toxicity level in text data, useful for moderating online content."
)
model_card.model_details.owners = [
    Owner(name='Project Pythia Team')
]
model_card.model_details.version.name = str(uuid.uuid4())
model_card.model_details.version.date = str(date.today())

# Ethical Considerations
model_card.considerations.ethical_considerations = [Risk(
    name='Potential biases in toxicity detection',
    mitigation_strategy='Regularly evaluate the model for fairness across different demographics and different languages and language groups'
)]

# Limitations (Placed correctly in the considerations section)
model_card.considerations.limitations = [Limitation(
    description='The model might show biases based on the training data and may not generalize well across all types of textual content.'
)]

# Use Cases
model_card.considerations.use_cases = [UseCase(
    description='The model is designed for identifying and moderating toxic content in multi-language text data.'
)]

# Intended Users
model_card.considerations.users = [User(
    description='The intended users are individuals and organizations who will use the Pythia API for content moderation and toxicity detection in various text-based platforms.'
)]

# Add links to evaluation data in the overview or another suitable section
model_card.model_details.overview += "\n\nEvaluation data and results can be found on the W&B platform: " \
                                     "[Training Project](https://wandb.ai/kizimayarik01/toxic_text_classification), " \
                                     "[Param Search Project](https://wandb.ai/kizimayarik01/toxicity_classification_sweep)."

# Update and export the model card
toolkit.update_model_card(model_card)
html_content = toolkit.export_format(output_file="model_card.html")

print("Model card created successfully. Check the `model_card.html` file in the output directory.")
