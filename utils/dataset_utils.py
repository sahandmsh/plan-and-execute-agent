from datasets import load_dataset


class DataSetUtil:
    """Utility for loading and sampling question-answering datasets.

    Attributes:
        None (stateless utility class).
    """

    def load_qa_dataset(
        self,
        dataset_name: str = "squad",
        samples_count: int = 5000,
        random_sampling: bool = False,
        dataset_split: str = "train",
    ):
        """Load and sample a QA dataset from Hugging Face.

        Args:
            dataset_name (str, optional): Name of the dataset to load. Defaults to "squad".
            samples_count (int, optional): Maximum number of samples to return. Defaults to 5000.
            random_sampling (bool, optional): If True, shuffle dataset before sampling. Defaults to False.
            dataset_split (str, optional): Dataset split (e.g., "train", "test", "validation"). Defaults to "train".

        Returns:
            list[dict]: Sampled items, each containing keys: 'id', 'question', 'context', 'answers'.
        """
        dataset = load_dataset(dataset_name, split=dataset_split)
        if random_sampling:
            dataset = dataset.shuffle()
        dataset = dataset.select(range(min(samples_count, len(dataset))))
        sampled_data = [
            {
                "id": item["id"],
                "question": item["question"],
                "context": item["context"],
                "answers": item["answers"],
            }
            for item in dataset
        ]
        return sampled_data
