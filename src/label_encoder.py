import json


class LabelEncoder:

    def __init__(self, class_names):
        self.class_names = list(class_names)

        self.label_to_index = {
            label: index
            for index, label in enumerate(self.class_names)
        }

        self.index_to_label = {
            index: label
            for label, index in self.label_to_index.items()
        }

    def encode(self, labels):

        encoded = []

        for label in labels:

            if label not in self.label_to_index:
                raise ValueError(
                    f"Unknown class label: {label}"
                )

            encoded.append(
                self.label_to_index[label]
            )

        return encoded

    def decode(self, indices):

        decoded = []

        for index in indices:

            if int(index) not in self.index_to_label:
                raise ValueError(
                    f"Unknown class index: {index}"
                )

            decoded.append(
                self.index_to_label[int(index)]
            )

        return decoded

    def save(self, output_path):

        mapping = {
            str(index): label
            for index, label in self.index_to_label.items()
        }

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                mapping,
                file,
                indent=4
            )