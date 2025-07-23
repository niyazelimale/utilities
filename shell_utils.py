# Function to print the progress status at same location in the terminal
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


#Function to split a string to fixed "n" number of fields.
#If fields are less than "n", then remaining fields wil be empty. If fields are more, those will be chopped off.
def split_to_n_fields(line, n, delimiter=','):
  fields = line.split(delimiter)
  #Pad with empty strings if less than n fields
  fields += [''] * (n - len(fields))
  #Truncate if more than n fields
  return fields[:n]

