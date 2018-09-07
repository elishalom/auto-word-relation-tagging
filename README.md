# Automatic Tagging of Word Relationships

## Installation

Before installing requirements, install graphviz
For mac using brew: 

    $ brew install graphviz

For installing the requirement, run the following command in `src` directory:
    
    $ pip install -r requirements.txt
    
If running outside of a virtual environment, run with `sudo`:

    $ sudo pip install -r requirements.txt
    
## Running

Querying the system is done by a command line. The command needs to be executed from the `src` directory. The input is a pair of words and the results is a set of keywords printed to the standard output. To query the words _germany_ and _berlin_ for example:

    $ python -m pairs_finder find germany berlin
    
The `python -m pairs_finder` executes the `pairs_finder` module. The `find` argument is the command being perform, finding keywords in this case. The last two arguments the words being queried.

The expected example will be a list of terms joined by `,`:

`city,capital,largest,known,located,centre,also,listen,one,national`

 
 
### First execution

During the first query to the system, some resources will be downloaded to the local machine automatically. Please allow it some time to finish downloading. Future executions are faster (~20 seconds).  