import streamlit as st
import os
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import cv2
import numpy as np


def process_video(video_path):
    # Reading video
    video_frames = read_video(video_path)

 # Reading video
    video_frames = read_video(video_path)

    # Initialising Tracker
    tracker = Tracker('models/best.pt')

    # Getting object
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')

    # Getting object positions
    tracker.add_position_to_tracks(tracks)

    # Adding camera movement estimator
    # Initialising by first frame
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')
    # Calling adjust camera position
    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks, camera_movement_per_frame)

    # Added View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolating/inserting ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Adding speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assigning player teams
    team_assigner = TeamAssigner()  # initialising team assigner
    # assigning team their colours and by giving them first frame and tracks of player in first frame
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])

    # Looping over each player in each frame and assigning them to colour team
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assigning ball to player function
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(
            player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            # we getting tracks of players of frame numbers and then assign teamp player with team
            team_ball_control.append(
                tracks['players'][frame_num][assigned_player]['team'])
        else:
            # last person who has the ball
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Calling draw output function for object tracks
    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control)

    # Drawing camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame)

    # Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(
        output_video_frames, tracks)

    # Save the processed video
    output_path = '/Users/aaronjoju/Documents/PlayVision/output_videos/output_video.mp4'
    save_video(output_video_frames, output_path)
    return output_path


# Streamlit UI
st.title("PlayVision Video Processor")
st.write("Upload a video to process and view the output.")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])

if uploaded_file is not None:
    # Save uploaded file locally
    video_path = os.path.join(
        "/Users/aaronjoju/Documents/PlayVision/input-videos", uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(video_path)

    # Process video
    with st.spinner("Processing video... This may take some time."):
        output_path = process_video(video_path)
        print(output_path)

    st.success("Video processing complete!")
    st.write("Processed video:")

    # Display output video
    st.video(output_path)