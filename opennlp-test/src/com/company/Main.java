package com.company;


import opennlp.tools.ngram.NGramModel;
import opennlp.tools.tokenize.SimpleTokenizer;
import opennlp.tools.util.StringList;

import java.util.*;


public class Main {
    public static void main(String[] args) {
        NGramMachine x = new NGramMachine();
        Set<String> z = x.extract("Hello, my name is Paco.", 3);
        z.addAll(x.extract("Zapdos smokes Pookie", 3));
        for (String elem : z)
            System.out.print(elem);
    }
}


class NGramMachine {
    public NGramMachine(){
    }

    public static HashSet<String> extract(String phrase, Integer n) {
        String tokens[] = SimpleTokenizer.INSTANCE.tokenize(phrase);
        NGramModel model = new NGramModel();
        model.add(new StringList(tokens), 1, n);
        Iterator iter = model.iterator();
        List<String> copy = new ArrayList<String>();
        while (iter.hasNext())
            copy.add(iter.next().toString());
        return new HashSet<String>(copy);
    }
}

