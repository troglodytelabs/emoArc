"""
Django management command to load book data from trajectory CSV.
"""
import csv
import os
from django.core.management.base import BaseCommand
from books.models import Book


class Command(BaseCommand):
    help = 'Load book data from trajectory CSV file'

    def add_arguments(self, parser):
        parser.add_argument(
            'csv_path',
            type=str,
            help='Path to the trajectories CSV file (or directory containing part-*.csv files)'
        )

    def handle(self, *args, **options):
        csv_path = options['csv_path']

        # Check if path is a directory (Spark output with part files)
        csv_files = []
        if os.path.isdir(csv_path):
            self.stdout.write(f"Loading from directory: {csv_path}")
            for filename in os.listdir(csv_path):
                if filename.startswith('part-') and filename.endswith('.csv'):
                    csv_files.append(os.path.join(csv_path, filename))
            csv_files.sort()
        else:
            csv_files = [csv_path]

        if not csv_files:
            self.stdout.write(self.style.ERROR(f"No CSV files found in {csv_path}"))
            return

        total_loaded = 0
        total_updated = 0
        total_errors = 0

        for csv_file in csv_files:
            self.stdout.write(f"\nProcessing: {csv_file}")

            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)

                    for row in reader:
                        try:
                            # Get or create book
                            book, created = Book.objects.get_or_create(
                                book_id=row['book_id'],
                                defaults={
                                    'title': row.get('title', ''),
                                    'author': row.get('author', ''),
                                }
                            )

                            # Update basic metadata
                            book.title = row.get('title', book.title)
                            book.author = row.get('author', book.author)

                            # Update emotion scores
                            for emotion in ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']:
                                value = row.get(f'avg_{emotion}', 0.0)
                                setattr(book, f'avg_{emotion}', float(value) if value else 0.0)

                            # Update VAD scores
                            for vad in ['valence', 'arousal', 'dominance']:
                                value = row.get(f'avg_{vad}', 0.0)
                                setattr(book, f'avg_{vad}', float(value) if value else 0.0)

                            # calculate dyad scores from basic emotions (plutchik's dyads)
                            # primary dyads (adjacent emotions on wheel)
                            book.avg_love = (book.avg_joy + book.avg_trust) / 2
                            book.avg_submission = (book.avg_trust + book.avg_fear) / 2
                            book.avg_alarm = (book.avg_fear + book.avg_surprise) / 2
                            book.avg_disappointment = (book.avg_surprise + book.avg_sadness) / 2
                            book.avg_remorse = (book.avg_sadness + book.avg_disgust) / 2
                            book.avg_contempt = (book.avg_disgust + book.avg_anger) / 2
                            book.avg_aggressiveness = (book.avg_anger + book.avg_anticipation) / 2
                            book.avg_optimism = (book.avg_anticipation + book.avg_joy) / 2

                            # secondary dyads (skip 1 emotion on wheel)
                            book.avg_guilt = (book.avg_joy + book.avg_fear) / 2
                            book.avg_curiosity = (book.avg_trust + book.avg_surprise) / 2
                            book.avg_despair = (book.avg_fear + book.avg_sadness) / 2
                            book.avg_unbelief = (book.avg_surprise + book.avg_disgust) / 2
                            book.avg_envy = (book.avg_sadness + book.avg_anger) / 2
                            book.avg_cynicism = (book.avg_disgust + book.avg_anticipation) / 2
                            book.avg_pride = (book.avg_anger + book.avg_joy) / 2
                            book.avg_hope = (book.avg_anticipation + book.avg_trust) / 2
                            book.avg_anxiety = (book.avg_anticipation + book.avg_fear) / 2
                            book.avg_outrage = (book.avg_surprise + book.avg_anger) / 2

                            # Update narrative arc metrics
                            book.num_chunks = int(row.get('num_chunks', 20))
                            book.rising_action_pct = float(row.get('rising_action_pct', 0.0)) if row.get('rising_action_pct') else 0.0
                            book.climax_position = float(row.get('climax_position', 0.0)) if row.get('climax_position') else 0.0
                            book.resolution_pct = float(row.get('resolution_pct', 0.0)) if row.get('resolution_pct') else 0.0
                            book.emotional_volatility = float(row.get('emotional_volatility', 0.0)) if row.get('emotional_volatility') else 0.0

                            # load topic modeling results (top 3 topics with words and probabilities)
                            # format: [{"topic_id": 3, "probability": 0.32, "words": ["love", "heart", ...]}, ...]
                            # only show topics with >10% probability to avoid noise from corpus-wide topics
                            topics = []
                            for i in [1, 2, 3]:
                                topic_id = row.get(f'top_topic_{i}')
                                topic_prob = row.get(f'top_topic_{i}_prob')
                                topic_words = row.get(f'top_topic_{i}_words', '')
                                if topic_id and topic_prob and float(topic_prob) > 0.10:  # only include if >10% probability
                                    try:
                                        topics.append({
                                            'topic_id': int(float(topic_id)),
                                            'probability': float(topic_prob),
                                            'words': topic_words.split(',') if topic_words else []
                                        })
                                    except (ValueError, TypeError):
                                        pass
                            book.dominant_themes = topics

                            # load emotion trajectory for arc charts
                            # stored as base64-encoded JSON string to avoid CSV parsing issues
                            trajectory_b64 = row.get('emotion_trajectory_json', '')
                            if trajectory_b64:
                                try:
                                    import json
                                    import base64
                                    # decode base64, then parse JSON
                                    json_str = base64.b64decode(trajectory_b64.encode('utf-8')).decode('utf-8')
                                    book.emotion_trajectory = json.loads(json_str)
                                except (json.JSONDecodeError, TypeError, ValueError):
                                    book.emotion_trajectory = []
                            else:
                                book.emotion_trajectory = []

                            book.save()

                            if created:
                                total_loaded += 1
                            else:
                                total_updated += 1

                            if (total_loaded + total_updated) % 100 == 0:
                                self.stdout.write(f"  Processed {total_loaded + total_updated} books...")

                        except Exception as e:
                            total_errors += 1
                            self.stdout.write(self.style.WARNING(f"  Error processing row: {e}"))
                            continue

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error reading file {csv_file}: {e}"))
                continue

        self.stdout.write(
            self.style.SUCCESS(
                f'\nSuccessfully loaded {total_loaded} books, updated {total_updated} books ({total_errors} errors)'
            )
        )
