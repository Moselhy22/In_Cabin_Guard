"""Face cropping utility for model input (v2.0 skeleton - Phase 2 deliverable)."""
class FaceCropper:
    """Extracts face ROI for MobileNetV3 input (224x224)."""
    def crop_largest_face(self, frame, target_size=(224, 224)):
        raise NotImplementedError("Phase 2: Implement face cropping pipeline")
    
    def get_face_bbox(self, frame):
        raise NotImplementedError("Phase 2: Implement bounding box extraction")
