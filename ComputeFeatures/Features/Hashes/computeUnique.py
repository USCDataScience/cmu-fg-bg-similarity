from computeHashFromHbase import tic_toc_print
hashesFile = '/memexdata/Dataset/processed/0004_IST/Features/SHA1/hashes.txt'
outFile =  '/memexdata/Dataset/processed/0004_IST/lists/Uniq_sha1.txt'

uniq_hashes = {}
isUniq = [] # T/F
parentId = [] # -1 if it's 1st unique (hence the parent)
children = []

lno = 1
with open(hashesFile) as f:
  for line in f:
    if line in uniq_hashes:
      isUniq.append(False)
      par = uniq_hashes[line]
      parentId.append(par)
      children[par - 1].append(lno)
    else:
      uniq_hashes[line] = lno
      isUniq.append(True)
      parentId.append(-1)
    children.append([])
    lno += 1
    tic_toc_print('Done for %d' % lno)

# All numbers written out to file will be 1-indexed as always
f = open(outFile, 'w')
print('Writing out')
for i in range(len(isUniq)):
  if not isUniq[i]:
    f.write('D %d\n' % parentId[i])
  else:
    f.write('U ')
    for el in children[i]:
      f.write('%d ' % el)
    f.write('\n')
f.close()

