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

                            # Update dyad scores (primary dyads)
                            for dyad in ['love', 'submission', 'alarm', 'disappointment', 'remorse', 'contempt', 'aggressiveness', 'optimism']:
                                value = row.get(f'avg_{dyad}', 0.0)
                                setattr(book, f'avg_{dyad}', float(value) if value else 0.0)

                            # Update secondary dyad scores
                            for dyad in ['guilt', 'curiosity', 'despair', 'unbelief', 'envy', 'cynicism', 'pride', 'hope', 'anxiety', 'outrage']:
                                value = row.get(f'avg_{dyad}', 0.0)
                                setattr(book, f'avg_{dyad}', float(value) if value else 0.0)

                            # Update narrative arc metrics
                            book.num_chunks = int(row.get('num_chunks', 20))
                            book.rising_action_pct = float(row.get('rising_action_pct', 0.0)) if row.get('rising_action_pct') else 0.0
                            book.climax_position = float(row.get('climax_position', 0.0)) if row.get('climax_position') else 0.0
                            book.resolution_pct = float(row.get('resolution_pct', 0.0)) if row.get('resolution_pct') else 0.0
                            book.emotional_volatility = float(row.get('emotional_volatility', 0.0)) if row.get('emotional_volatility') else 0.0

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
