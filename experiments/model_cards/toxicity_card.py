import model_card_toolkit as mct
from datetime import date
from model_card_toolkit import (
    ModelCardToolkit,
    Owner,
    Reference,
    Risk,
    Limitation,
    UseCase,
    User,
)
import uuid


def create_toxicity_model_card(
    model_name,
    owner_name,
    ethical_risk,
    ethical_mitigation,
    limitation_desc,
    use_case_desc,
    user_desc,
    eval_data_links,
    output_dir,
):
    """
    Create a model card for a toxicity classification model.

    Args:
    model_name (str): Name of the model.
    owner_name (str): Name of the model owner.
    ethical_risk (str): Description of potential ethical risks.
    ethical_mitigation (str): Strategies to mitigate ethical risks.
    limitation_desc (str): Description of model limitations.
    use_case_desc (str): Description of model use cases.
    user_desc (str): Description of intended users.
    eval_data_links (str): Links to evaluation data and results.
    output_dir (str): Directory to store the generated model card assets.

    Returns:
    str: Path to the generated HTML model card.
    """
    # Initialize the Model Card Toolkit
    toolkit = ModelCardToolkit(output_dir=output_dir)

    # Create a model card
    model_card = toolkit.scaffold_assets()

    # Basic Information
    model_card.model_details.name = model_name
    model_card.model_details.overview = f"This model predicts the toxicity level in text data, useful for moderating online content. {eval_data_links}"
    model_card.model_details.owners = [Owner(name=owner_name)]
    model_card.model_details.version.name = str(uuid.uuid4())
    model_card.model_details.version.date = str(date.today())

    # Ethical Considerations
    model_card.considerations.ethical_considerations = [
        Risk(
            name="Potential biases in toxicity detection",
            mitigation_strategy=ethical_mitigation,
        )
    ]

    # Limitations
    model_card.considerations.limitations = [Limitation(description=limitation_desc)]

    # Use Cases
    model_card.considerations.use_cases = [UseCase(description=use_case_desc)]

    # Intended Users
    model_card.considerations.users = [User(description=user_desc)]

    # Update and export the model card
    toolkit.update_model_card(model_card)
    html_file_path = toolkit.export_format(output_file="model_card.html")

    return f"Model card created successfully. Check the `{html_file_path}` file in the output directory."


# Example usage
# output_dir = "./experiments/model_cards/model_card_assets"
# eval_data_links = "Evaluation data and results can be found on the W&B platform: [Training Project](https://wandb.ai/kizimayarik01/toxic_text_classification), [Param Search Project](https://wandb.ai/kizimayarik01/toxicity_classification_sweep)."
# print(
#     create_toxicity_model_card(
#         model_name="Toxicity Classification Model",
#         owner_name="Project Pythia Team",
#         ethical_risk="Potential biases in toxicity detection",
#         ethical_mitigation="Regularly evaluate the model for fairness across different demographics and different languages and language groups",
#         limitation_desc="The model might show biases based on the training data and may not generalize well across all types of textual content.",
#         use_case_desc="The model is designed for identifying and moderating toxic content in multi-language text data.",
#         user_desc="The intended users are individuals and organizations who will use the Pythia API for content moderation and toxicity detection in various text-based platforms.",
#         eval_data_links=eval_data_links,
#         output_dir=output_dir,
#     )
# )
