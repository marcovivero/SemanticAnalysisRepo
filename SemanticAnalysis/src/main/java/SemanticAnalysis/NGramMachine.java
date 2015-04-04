package SemanticAnalysis;

import opennlp.tools.ngram.NGramModel;
import opennlp.tools.tokenize.SimpleTokenizer;
import opennlp.tools.util.StringList;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;

public class NGramMachine {
    public NGramMachine(){
    }

    public static HashSet<String> extract(String phrase, Integer n) {
        String tokens[] = SimpleTokenizer.INSTANCE.tokenize(phrase);
        NGramModel model = new NGramModel();
        model.add(new StringList(tokens), 1, n);
        Iterator iter = model.iterator();
        List<String> copy = new ArrayList<>();
        while (iter.hasNext())
            copy.add(iter.next().toString());
        return new HashSet<>(copy);
    }
}