package SemanticAnalysis;

import opennlp.tools.ngram.NGramModel;
import opennlp.tools.tokenize.SimpleTokenizer;
import opennlp.tools.util.StringList;

import java.util.*;

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
