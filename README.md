# MessageSpamDetection
A simple text analysis project. It will take less than 30 seconds to run the whole project.

There are several functions computing different characteristics of the message, including:
1. whether a message has links;
2. whether has spam words;
3. TF/IDF vector (this feature data was too large to upload)
4. message length;
5. whether includes a phone number;

Pandas package was used to manipulate and store the features metrix data.

The generated features were then put into wika to decide whether they are strong enough to decide whether a message is a spam or ham. Based on that, an if/else block was built to do a message detection, with 97.49% accuracy.
