import sys
import time

def update_counter_progress(total):
  for I in range(1, total+1):
    sys.stdout.write(f’\rProgress: {i}/{total}’)
    sys.stdout.flush()
    time.sleep(2)
  print() # Move to the next line after completion

if __name__ == “__main__”:
  update_counter_progress(10)
