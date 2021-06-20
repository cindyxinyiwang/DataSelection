langs = ["aze", "bel", "glg", "slk", "tur", "rus", "por", "ces"]
ddir = "data/ted_processed/"
out_src = "{}/corresponding_src".format(ddir)
out_trg = "{}/corresponding_trg".format(ddir)

trg2src_dicts = {}
for i, lan in enumerate(langs):
    src = "{}/{}_eng/ted-train.spm8000.{}".format(ddir, lan, lan)
    trg = "{}/{}_eng/ted-train.spm8000.eng".format(ddir, lan)
    trg_file = open(trg, "r")
    src_file = open(src, "r")
    for src_line, trg_line in zip(src_file, trg_file):
        src_line = src_line.strip()
        trg_line = trg_line.strip()
        if trg_line not in trg2src_dicts:
            trg2src_dicts[trg_line] = [None for _ in range(len(langs))]
        trg2src_dicts[trg_line][i] = src_line

src_out_file = open(out_src, 'w')
trg_out_file = open(out_trg, 'w')
for trg, src_lines in trg2src_dicts.items():
    trg_out_file.write(trg + '\n')
    for s_line in src_lines:
        if s_line is None:
            src_out_file.write('\n')
        else:
            src_out_file.write(s_line + '\n')

    src_out_file.write('EOF\n')
