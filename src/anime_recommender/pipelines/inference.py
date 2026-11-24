"""Session-based inference API. Profile acquisition remains explicit by nickname."""
from anime_recommender.inference.profile_session import ProfileSession
from anime_recommender.inference.model_versions import load_version_workspace
from anime_recommender.profile_service import profile_from_history


def create_session(root, **options):
    return ProfileSession(root, **options)


def recommend_profile(root, fetched_history, *, model_version=None, top_k=15):
    """Already-normalized history/requests; zero network, fit or discovery."""
    workspace = load_version_workspace(root, model_version)
    return profile_from_history(workspace, fetched_history, top_k)
