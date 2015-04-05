package com.company.java;


import opennlp.tools.ngram.NGramModel;
import opennlp.tools.tokenize.SimpleTokenizer;
import opennlp.tools.util.StringList;

import java.util.*;


public class Main {

    public static void main(String[] args) {
        HashMap<String, Integer> x = NGramMachine.extract("Hello , Hello", 2);
        HashMap<String, Integer> y = NGramMachine.extract("Hello , Goodbye", 2);
        List<HashMap<String, Integer>> copy = new ArrayList();
        copy.add(x);
        copy.add(y);
        LinkedHashSet<String> universe = NGramMachine.create_universe(copy.iterator());
        for (String e : universe) {
            System.out.print(e + " ");
        }
        Double[] zx = NGramMachine.hash2Vect(x, universe);
        Double[] zy = NGramMachine.hash2Vect(y, universe);
        for (Double e : zy) {
            System.out.print(e.toString() + " ");
        }
    }
}


class NGramMachine {
    public static HashMap<String, Integer> extract(String phrase, Integer n) {
        String tokens[] = SimpleTokenizer.INSTANCE.tokenize(phrase);
        NGramModel model = new NGramModel();
        model.add(new StringList(tokens), 1, n);
        Iterator<StringList> iter = model.iterator();
        HashMap<String, Integer> hashMap = new HashMap<String, Integer>();
        while (iter.hasNext()) {
            StringList x = iter.next();
            hashMap.put(x.toString(), model.getCount(x));
        }
        return hashMap;
    }

    public static LinkedHashSet<String> create_universe(Iterator<HashMap<String, Integer>> iter) {
        LinkedHashSet<String> universe = new LinkedHashSet<String>();
        while (iter.hasNext()) {
            HashMap<String, Integer> x = iter.next();
            for (String e : x.keySet()) {
                universe.add(e);
            }
        }
        return universe;
    }

    public static Double[] hash2Vect (
            HashMap<String, Integer> ngrams,
            LinkedHashSet<String> universe
    ) {
        Double[] vector = new Double[universe.size()];
        Integer count = 0;
        Set<String> keySet = ngrams.keySet();
        for (String e : universe) {
            if (keySet.contains(e)) {
                vector[count] = ngrams.get(e).doubleValue();
            } else {
                vector[count] = 0.0;
            }
            count += 1;
        }
        return vector;
    }
}
