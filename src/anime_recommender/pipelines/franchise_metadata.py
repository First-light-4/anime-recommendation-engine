"""Explicit metadata operation; never invoked by recommendation inference."""
from anime_recommender.ingestion.viewing_order_metadata import run_metadata_pipeline


def run_franchise_metadata(root, config=None, *, execute=False, allow_network=False,
                           selected_mal_ids=None, **options):
    if execute and not allow_network:
        raise ValueError("Metadata update requires explicit allow_network=True")
    return run_metadata_pipeline(root, config, execute=execute,
                                 selected_mal_ids=selected_mal_ids, **options)
