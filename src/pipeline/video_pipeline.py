"""
Video Pipeline Module
Executes frame-by-frame video processing.
Models are loaded EXACTLY ONCE upon initialization.
Uses H.264 encoding via imageio-ffmpeg for universal web-browser playback.
"""
import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple

from src.pipeline.image_pipeline import ImagePipeline

class VideoPipeline:
    """
    Video processing pipeline that reuses pre-loaded models across all frames.
    """

    def __init__(self):
        # Models loaded ONCE here
        self.image_pipeline = ImagePipeline()

    def process_video(
        self,
        video_input_path: str,
        output_video_path: str = "output_analyzed_video.mp4",
        target_fps: float = None,
        frame_stride: int = 1
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Processes video frame by frame and writes web-compatible H.264 output video.

        Args:
            video_input_path (str): Path to input video file
            output_video_path (str): Path to write annotated video
            target_fps (float): Desired output FPS (defaults to input FPS)
            frame_stride (int): Process every N-th frame for speed optimization (default 1)

        Returns:
            tuple: (output_video_path, overall_video_json_report)
        """
        cap = cv2.VideoCapture(video_input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_input_path}")

        in_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        in_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        in_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if in_fps <= 0 or np.isnan(in_fps):
            in_fps = 25.0

        out_fps = target_fps if target_fps else (in_fps / max(1, frame_stride))

        os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)

        # Setup Web-Compatible H.264 Writer via imageio-ffmpeg or OpenCV fallback
        writer = None
        use_imageio = False
        try:
            import imageio
            writer = imageio.get_writer(
                output_video_path,
                fps=out_fps,
                codec='libx264',
                quality=6,
                format='FFMPEG',
                pixelformat='yuv420p'
            )
            use_imageio = True
        except Exception as e:
            print(f"[WARN] imageio-ffmpeg writer fallback to OpenCV: {e}")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, out_fps, (in_width, in_height))
            use_imageio = False

        frame_index = 0
        processed_frames_count = 0
        frame_reports = []
        hazard_scores = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_stride == 0:
                # Process single frame using pre-loaded pipeline
                annotated_frame_bgr, frame_report = self.image_pipeline.process(
                    image=frame,
                    source_name=f"{os.path.basename(video_input_path)}_frame_{frame_index}"
                )

                if use_imageio:
                    # imageio expects RGB format
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)
                    writer.append_data(annotated_frame_rgb)
                else:
                    writer.write(annotated_frame_bgr)

                frame_reports.append(frame_report)
                h_score = frame_report.get("summary", {}).get("road_hazard_indicator")
                if h_score is not None:
                    hazard_scores.append(h_score)
                processed_frames_count += 1

            frame_index += 1

        cap.release()
        if use_imageio:
            writer.close()
        else:
            writer.release()

        avg_hazard = int(np.mean(hazard_scores)) if hazard_scores else 0
        max_hazard = int(np.max(hazard_scores)) if hazard_scores else 0

        video_summary_report = {
            "project": "Indian Road Intelligence System",
            "video_source": os.path.basename(video_input_path),
            "total_frames_in_video": total_frames,
            "processed_frames_count": processed_frames_count,
            "input_resolution": f"{in_width}x{in_height}",
            "output_fps": round(out_fps, 2),
            "video_hazard_summary": {
                "average_road_hazard_indicator": avg_hazard,
                "peak_road_hazard_indicator": max_hazard,
                "overall_hazard_rating": "HIGH" if avg_hazard >= 55 else ("MODERATE" if avg_hazard >= 25 else "LOW")
            },
            "component_statuses": self.image_pipeline.get_component_statuses(),
            "frame_by_frame_sample_reports": frame_reports[:10]
        }

        return output_video_path, video_summary_report
