package com.rakib.deeplearning.service;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

public class CSVReader {
    public static List<CSVRecord> parse(File file) throws IOException{
        CSVParser parser = CSVParser.parse(file, StandardCharsets.UTF_8, CSVFormat.newFormat(','));
        return parser.getRecords();
    }
}
