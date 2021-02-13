import nltk
import sys
import codecs #permette di gestire file con diverse codifiche
import operator #permette di fare operazioni matematiche
import math # il modulo contiene le funzioni matematiche non direttamente supportate da Python

#Caricamento del modello statistico
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#Funzione che estrae e restituisce le frasi del testo
def getPhrases(file):
    phrases = tokenizer.tokenize(file)
    return phrases

#Funzione che calcola e restituisce il numero delle frasi del testo
def getCountPhrases(phrases):
    count = len(phrases)
    return count

#Funzione che calcola e restituisce il numero di tokens del testo 
def getTokens(phrases):
    tokenList = []   
    for phrase in phrases: #estraggo i token di ogni frase mediante la funzione fornitami da nltk. Utilizzando il ciclo for, aggiungo i token di ogni frase in una lista
        tokenSinglePhrase = nltk.word_tokenize(phrase)
        tokenList+=tokenSinglePhrase
    return tokenList

#funzione che calcola e restituisce il numero totale di token
def getCountTokens(token):
    count = len(token)
    return count

#Funzione che restituisce la lunghezza media delle frasi in termini di token
def getAverageLenTokens(phrases,token):
    averageTok = token / phrases #divido il numero totale di token con il numero totale di frasi
    return averageTok

#Funzione che calcola e restituisce il numero di caratteri del libro
def getCharacters(tokens):
    charactersTot= 0
    for token in tokens: #il ciclo for mi permette di calcolare il numero di caratteri per ogni token e di aggiungerlo, di volta in volta, in una variabile.
        characterSingleToken = len(token)
        charactersTot+=characterSingleToken
    return charactersTot

#Funzione che calcola e restituisce la lunghezza media delle parole in termini di caratteri
def getAverageLenCharacters(token,caratteri):
    averageChar= caratteri/token #divido il numero totale di caratteri con il numero totale di token
    return averageChar

#Funzione che calcola e restituisce il numero delle parole tipo all'aumentare del corpus per porzioni incrementali di 1000 token
def getVocabulary(tokens,index):
    tokensTot = []
    for tok in range (0,index):
        if tok < index:   #il ciclo for controlla che il numero di token sia inferiore a quello dell'index e li aggiunge ad una lista
            tokensTot.append(tokens[tok])
    typeWords = set(tokensTot) # la funzione set mi permette di ottenere la lista degli elementi diversi presenti in "tokensTot"
    lenTypeWords = len(typeWords)
    return lenTypeWords

#Funzione che estrae e restituisce il numero di hapax presenti nel testo
def getHapax(tokens):
    hapaxTot = 0
    for tok in tokens: # il ciclo for controlla la frequenza di ogni token, considerando soltanto quelli con frequenza pari a 1
        freqT = tokens.count(tok)
        if freqT == 1:
            hap = freqT
            hapaxTot += hap
    return hapaxTot

#Funzione che calcola e restituisce la distribuzione degli hapax all'aumentare del corpus per porzioni incrementali di 1000 token
def getHapaxDistrib(tokens,index):
    v1 = getHapax(tokens) / index
    return v1

#Funzione che estrae e restituisce la Pos di ogni token
def getPosTokens(tokenList):
    posT = nltk.pos_tag(tokenList) #funzione prende come parametro la lista dei token e restituisce una lista di bigrammi (token,pos)
    return posT

#Funzione che estrae e restituisce il numero dei sostantivi a partire dalla lista di bigrammi(token,pos)
def getNouns(bigrams):
    nounsList = []
    for pos in bigrams: # con il ciclo for controllo la pos dei diversi bigrammi, se è uguale alla pos di un sostantivo la aggiunge alla lista
        if pos[1] in ['NN','NNS','NNP','NNPS']:
            nounsList.append(pos)
    countNouns = len (nounsList) 
    return countNouns

#Funzione che estrare e restituisce il numero dei verbi a partire dalla lista di bigrammi (token,pos)
def getVerb(bigrams):
    verbs = []
    for pos in bigrams: #il ciclo for controlla la pos dei diversi bigrammi, se è uguale alla pos di un verbo la aggiunge ad una lista
        if pos[1] in ['VB','VBD','VBG','VBN','VBP','VBZ']:
            verbs.append(pos)
    countVerbs = len(verbs) 
    return countVerbs

#Funzione che calcola e restituisce il rapporto tra sostantivi e verbi
def getDivision (nouns,verbs):
    divisionNv = nouns / verbs
    return divisionNv

#Funzione che estrae e restituisce le Pos
def getPosList (bigrams):
    posB = []
    for pos in bigrams: # per ogni bigramma, aggiunge le Pos in una lista
        posB.append(pos[1])
    return posB

#Funzione che calcola e restituisce le 10 Pos più frequenti 
def getFreqPos (pos):
    freqPos = nltk.FreqDist(pos) #calcolo la distribuzione di frequenza delle pos
    posFreq = freqPos.most_common(10) #estraggo i 10 elementi più frequenti
    return posFreq
    

#Funzione che restituisce i bigrammi di Pos
def getPosBigrams (pos):
    posBigrams = nltk.bigrams(pos)
    return list(posBigrams) #trasformo l'oggetto bigrams in una lista mediante la funzione list()

#Funzione che trasforma e restituisce la lista dei bigrammi Pos senza ripetizioni
def getSetPosBigrams(pos):
    setPos = set(pos)
    return setPos 

#Funzione che calcola e restituisce i 10 bigrammi PoS con probabilità condizionata massima
def getProbCondMax(posBigrams,setPos,pos):
    posProb = []
    for posBigr in setPos:
        freqFirstEl = pos.count(posBigr[0]) #calcolo la frequenza del primo elemento della coppia PoS
        freqBigrams = posBigrams.count(posBigr) #calcolo la frequenza del bigramma PoS dalla lista dei bigrammi senza ripetizioni
        probMax = freqBigrams*1.0/freqFirstEl*1.0 # divido la frequenza del bigramma per la frequenza del primo elemento, entrambi moltiplicati per 1.0 per ottenere un risultato con la virgola
        tupla = [] # creo una lista vuota in cui aggiungo sia il bigramma Pos sia la relativa probabilità condizionata
        tupla.append(posBigr)
        tupla.append(probMax)
        tupla = tuple(tupla) #creo una tupla
        posProb.append(tupla) #aggiungo la tupla alla mia lista posProb
    posProb.sort(key=operator.itemgetter(1), reverse = True) #ordino la lista secondo la frequenza decrescente
    return posProb[:10] #mi restituisce la probabilità condizionata massima di 10 bigrammi Pos

#Funzione che calcola e restituisce la probabilità di un singolo elemento
def getProbSingleElement(pos, posList):
    freqAss = posList.count(pos) #conto la frequenza assoluta di ogni Pos
    freqRelativa = (freqAss*1.0) / (len(posList)*1.0) #calcolo la probabilità dividendo la frequenza assoluta per il corpus
    return freqRelativa

#Funzione che calcola e restituisce la probabilità congiunta
def getProbabilityCong(bigram,posListBigrams,tokens):
    freqAssBigram = posListBigrams.count(bigram) #calcolo la frequenza assoluta del bigramma
    freqRelBigram = (freqAssBigram*1.0)/ (len(tokens)*1.0) #calcolo la probabilità del bigramma 
    return freqRelBigram

#Funzione che calcola e restituisce i 10 Pos con LMI massima
def getLMI(posListBigrams,posList,posSetBigrams):
    posLMI = []
    for posBigr in posSetBigrams:
        freqAssBigram = posListBigrams.count(posBigr) #calcolo la frequenza assoluta del bigramma Pos
        probFirstElem = getProbSingleElement(posBigr[0],posList) #calcolo la probabilità del primo elemento del bigramma
        probSecondElem = getProbSingleElement(posBigr[1],posList) #calcolo la probabilità del secondo elemento del bigramma
        probCong = getProbabilityCong(posBigr,posListBigrams,posList) #calcolo la probabilità congiunta del bigramma Pos
        MI = math.log2(probCong/(probFirstElem*probSecondElem)) # calcolo la MI
        LMI = (freqAssBigram) * (MI) # calcolo la local mutual information
        tupla = []
        tupla.append(posBigr)
        tupla.append(LMI)
        tupla = tuple(tupla)
        posLMI.append(tupla) #creo una tupla composta da bigramma PoS e LMI
    posLMI.sort(key=operator.itemgetter(1), reverse = True) #ordino la lista secondo frequenza decrescente
    return posLMI[:10] #restituisce i 10 Pos con LMI massima






def main(file1,file2):

    print ("\nStudente: Silvia Cuozzo , Matricola: 587958\n")
    print ("\nPrimo Libro: THE SECRET ADVERSARY By Agatha Christie\n")
    print ("\nSecondo Libro: THE VALLEY OF FEAR By Sir Arthur Conan Doyle\n")

    #Apertura dei file ricevuti in input dal programma
    fileInput1= codecs.open(file1,"r","utf-8")
    fileInput2 = codecs.open(file2,"r","utf-8")

    #Lettura dei due file di testo
    fileRead1 = fileInput1.read()
    fileRead2 = fileInput2.read()

    #Punto 1(a) : Numero totale di frasi e di token per ciascun libro
    phrasesFile1 = getPhrases(fileRead1) #richiamo la funzione per ottenere le frasi del primo libro
    phrasesFile2 = getPhrases(fileRead2) #richiamo la funzione per ottenere le frasi del primo libro
    countPhrasesFile1 = getCountPhrases(phrasesFile1) #richiamo la funzione per ottenere il numero di frasi del primo libro
    countPhrasesFile2 = getCountPhrases(phrasesFile2) #richiamo la funzione per ottenere il numero di frasi del secondo libro
    print("\n\n\n-Il numero totale di frasi:")
    print ("\nPrimo libro: ", countPhrasesFile1)
    print("\nSecondo libro: ", countPhrasesFile2)

    tokensFile1 = getTokens(phrasesFile1) #richiamo la funzione per ottenere i token del primo libro
    tokensFile2 = getTokens(phrasesFile2) #richiamo la funzione per ottenere i token del secondo libro
    countTokensFile1 = getCountTokens(tokensFile1) #richiamo la funzione per ottenere il numero di token del primo libro
    countTokensFile2 = getCountTokens(tokensFile2) #richiamo la funzione per ottenere il numero di token del secondo libro
    print("\n\n-Il numero totale di token:")
    print ("\nPrimo libro: ", countTokensFile1)
    print ("\nSecondo libro: ", countTokensFile2)

    #Punto 2(a): Lunghezza media delle frasi in termini di token e la lunghezza media delle parole in termini di caratteri
    AverageLenTokensFile1= getAverageLenTokens(countPhrasesFile1,countTokensFile1) #richiamo la funzione per ottenere la lunghezza media delle frasi in termini di token del primo libro
    AverageLenTokensFile2 = getAverageLenTokens(countPhrasesFile2, countTokensFile2)#richiamo la funzione per ottenere la lunghezza media delle frasi in termini di token del secondo libro
    print("\n\n\n-La lunghezza media delle frasi in termini di token:")
    print ("\nPrimo libro: ", AverageLenTokensFile1)
    print ("\nSecondo libro: ", AverageLenTokensFile2)

    characters1 = getCharacters(tokensFile1) #richiamo la funzione per ottenere il numero dei caratteri del primo libro
    characters2 = getCharacters(tokensFile2) #richiamo la funzione per ottenere il numero dei caratteri del secondo libro
    averageLenChar1 = getAverageLenCharacters(countTokensFile1,characters1) #richiamo la funzione per ottenere la lunghezza media dei token in termini di caratteri del primo libro
    averageLenChar2 = getAverageLenCharacters(countTokensFile2,characters2) #richiamo la funzione per ottenere la lunghezza media dei token in termini di caratteri del secondo libro
    print("\n\n\n-La lunghezza media delle parole in termini di caratteri:")
    print ("\nPrimo libro: ",averageLenChar1)
    print ("\nSecondo libro: ",averageLenChar2)

    #Punto 3(a): la grandezza del vocabolario all'aumentare del corpus per porzioni incrementali di 1000 token
    print("\n\n\n-La grandezza del vocabolario:")
    for tok in range (1000,69001,1000): # indico gli intervalli di token che prendo in considerazione
        lenVocabularyFile1 = getVocabulary(tokensFile1,tok) #richiamo la funzione per ottenere la lunghezza del vocabolario del primo libro
        print ("\nNumero di parole tipo del primo libro a ", tok, " tokens è:", lenVocabularyFile1)
        lenVocabularyFile2 = getVocabulary(tokensFile2,tok) #richiamo la funzione per ottenere la lunghezza del vocabolario del secondo libro
        print ("\nNumero di parole tipo del secondo libro a ", tok, " tokens è:", lenVocabularyFile2,"\n")

    #Punto 3(b):la distribuzione degli hapax all'aumentare del corpus per porzioni incrementali di 1000 token 
    print("\n\n\n-La distribuzione degli hapax:")
    for hapax in range(1000,69001,1000): #indico gli intervalli di token che prendo in considerazione
        hapaxDistrFile1 = getHapaxDistrib(tokensFile1,hapax) #richiamo la funzione per ottenere la distribuzione degli hapax del primo libro
        print ("\nPrimo libro a ", hapax, " tokens è:",hapaxDistrFile1)
        hapaxDistrFile2 = getHapaxDistrib(tokensFile2,hapax) #richiamo la funzione per ottenere la distribuzione degli hapax del secondo libro
        print ("\nSecondo libro a ", hapax , " tokens è:", hapaxDistrFile2, "\n")
    
    #Punto 4(a): il rapporto tra Sostantivi e Verbi
    posTokenFile1 = getPosTokens(tokensFile1) #richiamo la funzione per ottenere la Pos dei token del primo libro
    nounFile1 = getNouns(posTokenFile1) #richiamo la funzione per ottenere il numero delle Pos equivalente ai sostantivi
    verbsFile1 = getVerb(posTokenFile1) #richiamo la funzione per ottenere il numero delle Pos equivalente ai verbi
    divisionSv1 = getDivision(nounFile1,verbsFile1)
    print("\n\n\n-Il rapporto tra sostantivi e verbi:")
    print ("\nPrimo libro:", divisionSv1)
    posTokenFile2 = getPosTokens(tokensFile2) #richiamo la funzione per ottenere la Pos dei token del secondo libro
    nounFile2 = getNouns(posTokenFile2) #richiamo la funzione per ottenere il numero delle Pos equivalente ai sostantivi
    verbsFile2= getVerb(posTokenFile2) #richiamo la funzione per ottenere il numero delle Pos equivalente ai verbi
    divisionSv2 = getDivision(nounFile2,verbsFile2)
    print ("\nSecondo libro:", divisionSv2)
    
    #Punto 5(a): le 10 PoS (Part-of-Speech) più frequenti
    posListFile1 = getPosList(posTokenFile1) #richiamo la funzione per ottenere la lista delle Pos del primo libro
    freqPosFile1 = getFreqPos(posListFile1) #richiamo la funzione per ottenere le 10 Pos più frequenti del primo libro
    print ("\n\n\n-Le 10 Pos più frequenti del primo libro sono:")
    for element in freqPosFile1: #ciclo tutti gli elementi del bigramma
        print("\nPos: ", element[0], " con frequenza: ", element[1]) #stampo la Pos e la relativa frequenza

    posListFile2 = getPosList(posTokenFile2) #richiamo la funzione per ottenere la lista delle Pos del secondo libro
    freqPosFile2 = getFreqPos(posListFile2) #richiamo la funzione per ottenere le 10 Pos più frequenti del secondo libro
    print ("\n\n\n-Le 10 Pos più frequenti del secondo libro sono:")
    for element in freqPosFile2: #ciclo gli elementi del bigramma
        print("\nPos: ", element[0], " con frequenza: ", element[1]) #stampo la Pos e la relativa frequenza
    
    #Punto 6(a): Estrazione ed ordinamento dei 10 bigrammi di PoS con probabilità condizionata massima
    posBigramsFile1 = getPosBigrams (posListFile1) #richiamo la funzione per ottenere la lista dei Bigrammi Pos del primo libro
    posSetBigramsFile1 = getSetPosBigrams (posBigramsFile1) #richiamo la funzione per ottenere la lista dei Bigrammi Pos senza ripetizioni del primo libro
    posProbCondMaxFile1 = getProbCondMax (posBigramsFile1,posSetBigramsFile1,posListFile1) #richiamo la funzione per calcolare la probabilità condizionata massima
    print ("\n\n\n-I 10 bigrammi di Pos del primo libro con probabilità condizionata massima sono:")
    for element in posProbCondMaxFile1: #ciclo gli elementi del bigramma
        print("\nBigramma Pos: ", element[0], " con probabilità condizionata: ", element[1]) #stampo i bigrammi Pos e la relativa probabilità condizionata

    posBigramsFile2 = getPosBigrams (posListFile2) #richiamo la funzione per ottenere la lista dei Bigrammi Pos del secondo libro
    posSetBigramsFile2 = getSetPosBigrams (posBigramsFile2) #richiamo la funzione per ottenere la lista dei Bigrammi Pos senza ripetizioni del secondo libro
    posProbCondMaxFile2 = getProbCondMax (posBigramsFile2,posSetBigramsFile2,posListFile2) #richiamo la funzione per calcolare la probabilità condizionata massima 
    print ("\n\n\n-I 10 bigrammi di Pos del secondo libro con probabilità condizionata massima sono:")
    for element in posProbCondMaxFile2: #ciclo gli elementi del bigramma
        print("\nBigramma Pos: ", element[0], " con probabilità condizionata: ", element[1]) #stampo i bigrammi Pos e la relativa probabilità condizionata

    #Punto6(b): Estrazione ed ordinamento dei 10 bigrammi di Pos con forza associativa massima 
    LMIFile1 = getLMI(posBigramsFile1,posListFile1,posSetBigramsFile1) #richiamo la funzione per ottenere la LMI massima dei 10 bigrammi di Pos del primo libro
    print ("\n\n\n-I 10 bigrammi di Pos del primo libro con forza associativa massima sono:")
    for element in LMIFile1: #ciclo gli elementi del bigramma
        print("\nBigramma Pos: ", element[0], " con LMI: ", element[1]) #stampo i bigrammi Pos e la relativa LMI massima

    LMIFile2 = getLMI(posBigramsFile2,posListFile2,posSetBigramsFile2) #richiamo la funzione per ottenere la LMI massima dei 10 bigrammi di Pos del secondo libro
    print ("\n\n\n-I 10 bigrammi di PoS del secondo libro con forza associativa massima sono:")
    for element in LMIFile2: #ciclo gli elementi del bigramma
        print("\nBigramma Pos: ", element[0], " con LMI: ", element[1]) #stampo i bigrammi Pos e relativa LMI massima

main(sys.argv[1],sys.argv[2])
