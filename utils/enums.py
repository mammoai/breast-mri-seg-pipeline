from enum import StrEnum


class Laterality(StrEnum):
    """Laterality of the image (0020,0060). Adding "bilateral" for MRIs that contain
    both breasts.
    """

    LEFT = "left"
    RIGHT = "right"
    BILATERAL = "bilateral"

    @classmethod
    def from_image_laterality(cls, image_laterality: str):
        if isinstance(image_laterality, str):
            if image_laterality.lower() == "l":
                return cls("left")
            elif image_laterality.lower() == "r":
                return cls("right")
        return None


class View(StrEnum):
    """
    View of the image.
    """

    AX = "axial"
    SAG = "sagittal"
    COR = "coronal"
    MLO = "mediolateral_oblique"
    CC = "craniocaudal"

    @classmethod
    def from_view_position(cls, view_position: str):
        if isinstance(view_position, str):
            if view_position in ["AX", "SAG", "COR", "MLO", "CC"]:
                return getattr(cls, view_position)
        return None
