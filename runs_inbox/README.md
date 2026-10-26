# runs_inbox

Put the results_<round>_s<N>.zip files downloaded from Kaggle here, then run
`python3 scripts/kaggle_queue.py ingest`. Ingest unzips them into the repo, moves each zip to
`outputs/_zips_archive/` and records it in `outputs/_ingested.txt`; ingesting the same zip twice does nothing.
