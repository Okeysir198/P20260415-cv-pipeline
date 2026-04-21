"""
Class mapping utilities.

Maps source class names to canonical target class names.
Different datasets use different class names; this normalizes them.
"""



class ClassMapper:
    """
    Map source class names to canonical target class names.

    Example:
        source_classes = ["Hardhat", "NO-Hardhat", "Person"]
        target_classes = ["person", "head_with_helmet", "head_without_helmet"]
        class_map = {"Hardhat": "head_with_helmet", "NO-Hardhat": "head_without_helmet"}

        mapper = ClassMapper(target_classes, class_map)
        target_id = mapper.get_target_id("Hardhat")  # Returns: 1
    """

    def __init__(
        self,
        target_classes: list[str],
        source_to_target_map: dict[str, str] | None = None
    ):
        """
        Initialize class mapper.

        Args:
            target_classes: List of canonical target class names (order defines IDs)
            source_to_target_map: Optional dict mapping source names to target names
        """
        self.target_classes = target_classes
        self.target_name_to_id = {name: i for i, name in enumerate(target_classes)}
        self.source_to_target_map = source_to_target_map or {}

    def get_target_id(self, source_class: str) -> int | None:
        """
        Map a source class name to target class ID.

        Args:
            source_class: Source class name

        Returns:
            Target class ID, or None if not found
        """
        # First apply source->target name mapping if configured
        if self.source_to_target_map and source_class in self.source_to_target_map:
            target_name = self.source_to_target_map[source_class]
        else:
            target_name = source_class

        # Then look up target ID
        return self.target_name_to_id.get(target_name)

    def get_target_name(self, source_class: str) -> str | None:
        """
        Map a source class name to canonical target class name.

        Args:
            source_class: Source class name

        Returns:
            Target class name, or None if not found
        """
        # Apply source->target name mapping if configured
        if self.source_to_target_map and source_class in self.source_to_target_map:
            return self.source_to_target_map[source_class]

        # Check if it's already a valid target name
        if source_class in self.target_name_to_id:
            return source_class

        return None

    def validate_mapping(self, source_classes: set[str]) -> list[str]:
        """
        Validate that all source classes have mappings.

        Args:
            source_classes: Set of source class names found in data

        Returns:
            List of unmapped class names (empty if all mapped)
        """
        unmapped = []
        for source_class in source_classes:
            target_name = self.get_target_name(source_class)
            if target_name is None:
                unmapped.append(source_class)
        return unmapped

