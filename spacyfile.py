import numpy as np
import spacy
from spacy import displacy
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from spacy.tokens import Span

text = """
    Clean Code ist ein Begriff aus der Softwaretechnik, der seinen Ursprung im gleichnamigen Buch von Robert Cecil Martin hat. 
    Als „sauber“ bezeichnen Softwareentwickler in erster Linie Quellcode, aber auch Dokumente, Konzepte, Regeln und Verfahren, 
    die intuitiv verständlich sind. Als intuitiv verständlich gilt alles, was mit wenig Aufwand und in kurzer Zeit richtig 
    verstanden werden kann. Vorteile von Clean Code sind stabilere und effizient wartbarere Programme, d. h. kürzere 
    Entwicklungszeiten bei Funktionserweiterung und Fehlerbehebungen. Die Bedeutung wächst mit der Beobachtung, 
    dass im Schnitt 80 % der Lebensdauer einer Software auf den Wartungszeitraum entfällt.
    Die Notwendigkeit, Code noch nach der Entwicklung von „unsauberen“ Stellen zu reinigen, wird häufig nicht 
    gesehen oder vom Management nicht bewilligt, sobald das Programm seine vorgesehene Funktion ausübt. 
    Ein direktes Schreiben von „sauberem“ Code ist nahezu unmöglich, kann jedoch durch den bewussten Umgang 
    mit den Prinzipien und Praktiken von Clean Code verbessert werden.
"""

new_text = """
                    Mal sehen was passiert wenn wir einen neuen Textabschnitt hinzufügen. Ich hoffe es funktioniert.
                """

text_2 = """
            Erweitert man die Anzahl der Texte passt sich die neue Verteilungsfunktion an. Das ganze ist so geschrieben, dass die extrahierten Sätze für weitere Analysen bereit stehen. Außerdem habe ich zum besseren Verständnis, Kommentare hinzugefügt.
            """

class SentencelengthCounter():  ## Class to compute a pdf over possible sentence lengths for a Text(String) as input

    def __init__(self, voc, text):
        # Initialize member variables
        if self.checkifvalidinput(text) == True:  # check if input is from type string
            self.input = text
            self.vocabular = voc  # String corresponding to german language
            self.nlp = spacy.load(voc)  # Load a natural language proccessing classifier (for german language)
            self.doc = self.nlp(text)  # get the analyzed document object from nlp classfier
            self.sents_ = self.doc.sents  # extract the classified sentences from doc object
            self.numofsents = 0
            self.counts = Counter()

    def trueSentence(self, sentence):
        if len(list(sentence)) == 1:  # sentences with only 1 character like ' ' are not valid sentences
            return False
        return True

    #### possibility to compute the sentencelength on different ways count (words, syllables, letters)
    #### here we use the definition: amount of words, but it's easy to extend for other definitions
    def sentencelength(self, sentence):  # return sentencelength (number of words in a sentence)
        senlen = 0
        for word in sentence:  # go through words in sentence object and proof if their from type alpha and not from type punct
            if word.is_alpha or not word.is_punct:  # without ' ', ','', '.' etc. (we just count words!)
                senlen += 1
        return senlen

    # return adjective for sentencelength (depending on the number of words in a sentence)
    def getsentencelengthtype(self, sentence):
        senlen = self.sentencelength(sentence)  # get the number of words in sentence
        if senlen < 10:
            return "short"
        elif senlen < 25:
            return "normal"
        else:
            return "long"

    def probabilitydensityfunc(self):  # create a  probabiltydensityfunction (dict type)
        sentencelengths = []  # list of sentencelengths for real sentences
        ## go through the doc (of extracted sentences) and collect the sentencelengths if extracted sentence is a 'real' sentence
        for sent in self.sents_:
            if self.trueSentence(sent):
                sentencelengths.append(self.sentencelength(sent))  # collect the sentencelengths
                self.numofsents += 1  # count sentences to normalize later

        self.counts.update(
            sorted(sentencelengths))  # use Counter object to count how often appears which sentencelength in the text

        """
            Transform the Counter object to a probabilitydensityfunc (type dictonary) 
            which depends on the sentencelenths
        """
        probdens = {}  ## dictonary {key: sentencelengths, value: likelihood of occurence}
        for senleng in self.counts:
            probdens[senleng] = self.counts[senleng] / self.numofsents  # normalize the values to get probability
        return probdens

    def getcounter(self):  # return Counterobject for the input text (which sentencenlength shows up how often)
        return self.counts

    def getnumofsents(self):  # return the number of sentences in a text
        return self.numofsents

    def checkifvalidinput(self, text):  # check if input is from type string
        if isinstance(text, str):
            return True
        else:
            print("Input is not a valid String")
            return False


class SpacySenEntVisualizer(SentencelengthCounter):
    def __init__(self, voc, text, textindex):
        super().__init__(voc, text)  # run constructor of SentencelengthCounter
        self.colors = {"long": "red", "normal": "green",
                       "short": "blue"}  # use colors to differentiate the entities better
        self.options = {"colors": self.colors}
        print(" #### Initialize Displacy Entities Visualizer #### \n")
        print("Displacy Visualizer for entities per sentence (Text {})".format(textindex))

    def visualize(self):
        offset_index = 0 # offset to extract the right parts of the text
        span_list = []
        for sen in self.sents_:  # go trough all sentences
            if self.trueSentence(sen):  # just take the 'real' sentences
                span = self.setlengthlabels_(sen, offset_index)  # get the span object with implemented entity labels to the corresponding sentence
                span_list.append(span)
                self.doc.set_ents([span], default='unmodified')  # add new entity
            offset_index += len(sen)  # increase the offset to adapt it for the next sentence

        displacy.serve(span_list, style="ent",
                       options=self.options)  # display sentence with entity (NER Visualizer)

    def setlengthlabels_(self, sen, offset_index):  # set for every sentence the corresponding entity label
        label = self.getsentencelengthtype(sen)  # get the corresponding label to the sentence
        span = Span(self.doc, start=offset_index, end=len(sen) + offset_index,
                    label=label)  # creat span object with 'label' as entity label
        return span


class pdfVisualizer():

    def __init__(self, probdens, maxsenlen):
        self.probdens = probdens
        self.maxsenlen = maxsenlen  ## max sentence length (just to limit the histogram plot)

    def getData(self):
        return list(self.probdens.keys()), list(self.probdens.values())

    def transformTopandas(self, data):
        return pd.DataFrame({'Sentencelength': data[0], 'Likelihood of occurrence': data[1]},
                            columns=['Sentencelength', 'Likelihood of occurrence'])

    def visualizeData(self):
        df = self.transformTopandas(self.getData())

        ### Transform into histogramm with max sentence legth
        ### Includes zero elements (just to visualize it better)
        histogram = pd.DataFrame({'Sentencelength': np.arange(0, self.maxsenlen + 1),
                                  'Likelihood of occurrence': np.zeros(self.maxsenlen + 1)},
                                 columns=['Sentencelength', 'Likelihood of occurrence'])
        for j in histogram.index:
            for i in df.index:
                if df['Sentencelength'][i] == histogram['Sentencelength'][j] and df['Sentencelength'][i] < self.maxsenlen:
                    histogram['Likelihood of occurrence'][j] = df['Likelihood of occurrence'][i]

        histogram.plot(x='Sentencelength', kind="bar", figsize=(9, 8),
                       title="Pobabilitydensityfunction over sentencelengths", legend='Likelihood of occurrence')
        plt.show()


def extendDistribution(Counters_):  # possible to extend distribution for many SentlenCounters (need a list)
    print("Combined pdf for all texts")
    pdf_combined = {}
    combined_counter = Counter()
    combined_numofsents = 0
    for Counter_ in Counters_:  # go through all Counters from all analyzed texts
        combined_counter.update(
            Counter_.getcounter())  # Add sentencelengths and their occurence to a Counter that includes all texts
        combined_numofsents += Counter_.getnumofsents()  # Count all number of sentences

    for senleng in sorted(combined_counter.keys()):
        pdf_combined[senleng] = combined_counter[
                                    senleng] / combined_numofsents  # normalize the values to get a probabilty
    return pdf_combined

def analyzesentencelengths(text, index, maxlen):

    SenlenCounter = SentencelengthCounter('de_core_news_sm', text)  # Initialize SentencelengthCounter per text
    probdens = SenlenCounter.probabilitydensityfunc()  # pdf over sentencelengths

    ### Visualize entities for every sentence with displacy
    spacyvisualizer = SpacySenEntVisualizer(SenlenCounter.vocabular, SenlenCounter.input,
                                                index + 1)  # create entity visualize object
    spacyvisualizer.visualize()  # visualize results

    ### Visualize the calculated pdf over the sentencelengths
    Sentencedistributionvisualizer = pdfVisualizer(probdens, maxlen)  # create the pdfVisualizer
    print("Dictonary: pdf over sentencelengths for Text {} \n".format(index + 1))
    print(Sentencedistributionvisualizer.transformTopandas(
            Sentencedistributionvisualizer.getData()))  # print pdf in pandas DataFrame type
    Sentencedistributionvisualizer.visualizeData()

    return SenlenCounter


def main(texts):
    """
       Want to write a programm that analyzises how often which sentencelength appears in a given text
       (or for a tuple of different texts to get a more global solution about sentencelengths distribution)
    """
    max_senlen = 35  ## max sentence length (just to limit the histogram plot)

    SentencelenCounterlist = []  # a list which will contain all SentencelenCounters for every text input

    for index, text_ in enumerate(texts):  # loop through all texts
        # analyze every text, visualize the results and collect the Sentencelengthcounters (to combine them finally)
        SentencelenCounterlist.append(analyzesentencelengths(text_, index, max_senlen))

    ## combine SentencelenCounters and visualize the new distribution ##
    print("##############\n")
    print("Now it's on to combine all pdf's to get one big pdf")
    print("Sentencelength Probabilitydensityfunction of all texts \n")
    print("##############\n")

    Sentencedistributionvisualizer_combined = pdfVisualizer(extendDistribution(SentencelenCounterlist), max_senlen)
    print(Sentencedistributionvisualizer_combined.transformTopandas(Sentencedistributionvisualizer_combined.getData()))
    Sentencedistributionvisualizer_combined.visualizeData()  # visualize the data with bars


if __name__ == "__main__":
    text_tuple = (text, new_text,
                  text_2)  ## if you want to analyze more than one text you can insert a tuple of text, there is also the possibilit to pass just one text but as a tuple () with one element
    """
        To analyze a text or more texts just pass the text/s as a tuple to the main. The function analyzes

    """
    main(text_tuple)
