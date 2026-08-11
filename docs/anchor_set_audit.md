# The twenty anchors under both configurations (milestone M23.10.3)

`python scripts/audit_anchor_set.py` — 2026-08-10. 20 anchors × 2 configurations × 10 slots = **400 slots** to read.

Configurations: **A** = Matrix factorization (ALS) · **B** = Item-based collaborative filtering. **Both sit at anchor floor 50** and the candidate floor stays 20 in both (L34), so the switch moves one variable.

`co` is the number of the anchor's readers who also read that book — the number the app prints. The `score` column is deliberately **not** here: A's is a cosine between learned profiles and B's a shrunk cosine over shared readers, they are not comparable, and a table that put them side by side would invite exactly the comparison M23 decision 6 exists to prevent.

Every anchor below was reached by typing the title into the app's own search box, not by work id — see the resolution table for what each query actually landed on.

## 1 · What each typed title resolves to

**2 of 20 do not land on the book the set names.** Both are read in section 3 where they differ: the book a visitor gets, because that is what the gate is for, and the book the set intended, because that is the one chosen to be judgeable.

| # | typed | resolved to | readers | ✓ | also offered |
|--:|---|---|--:|:-:|---|
| 1 | The Lovely Bones | The Lovely Bones: A Novel by *Alice Sebold* | 1,295 | ✓ | The Lovely Bones (103), The Bone People (51), BAG OF BONES : A NOVEL (79) |
| 2 | The Da Vinci Code | The Da Vinci Code by *Dan Brown* | 905 | ✓ | — |
| 3 | Harry Potter and the Sorcerer's Stone | Harry Potter and the Sorcerer's Stone by *J. K. Rowling* | 832 | ✓ | — |
| 4 | Bridget Jones's Diary | Bridget Jones's Diary by *Helen Fielding* | 772 | ✓ | The SECRET DIARY OF ANNE BOLEYN (52), The Diary of Ellen Rimbauer: My Life at Rose Red (96), Suzanne's Diary for Nicholas (440), The Forest House (78) |
| 5 | Girl with a Pearl Earring | Girl with a Pearl Earring by *Tracy Chevalier* | 647 | ✓ | — |
| 6 | Interview with the Vampire | Interview with the Vampire by *Anne Rice* | 521 | ✓ | Vittorio the Vampire: New Tales of the Vampires (113), The Vampire Lestat (Vampire Chronicles, Book II) (310), The Vampire Armand (The Vampire Chronicles, Book 6) (63), Circus of the Damned (Anita Blake Vampire Hunter (Paperback)) (80) |
| 7 | To Kill a Mockingbird | To Kill a Mockingbird by *Harper Lee* | 495 | ✓ | — |
| 8 | The Hobbit | The Hobbit : The Enchanting Prelude to The Lord of the Rings by *J.R.R. TOLKIEN* | 281 | ✓ | The Hobbit (Leatherette Collector's Edition) (123), The Hobbit: or There and Back Again (112) |
| 9 | Dune | Dune by *Frank Herbert* | 257 | ✓ | Children of Dune (Dune Chronicles, Book 3) (81), House of Sand and Fog (583), The Beach (102), In This Mountain (53) |
| 10 | Fight Club | Fight Club by *Chuck Palahniuk* | 102 | ✓ | — |
| 11 | Guns, Germs, and Steel | Secrets by *DANIELLE STEEL* | 98 | **✗** | Malice (95), Wanderlust (89), Guns, Germs, and Steel: The Fates of Human Societies (67), Palomino (64) |
| 12 | Life of Pi | Life of Pi by *Yann Martel* | 658 | ✓ | — |
| 13 | Tuesdays with Morrie | Tuesdays with Morrie: An Old Man, a Young Man, and Life's Greatest Lesson by *MITCH ALBOM* | 492 | ✓ | Paradise (111), The Day After Tomorrow (86), Morning Glory (62), A Season Beyond a Kiss (57) |
| 14 | Angela's Ashes | Angela's Ashes (MMP) : A Memoir by *Frank McCourt* | 326 | ✓ | ANGELA'S ASHES (283), Angela's Ashes: A Memoir (224) |
| 15 | Love in the Time of Cholera | Love in the Time of Cholera by *Gabriel Garcia Marquez* | 261 | ✓ | — |
| 16 | One Hundred Years of Solitude | One Hundred Years of Solitude by *Gabriel Garcia Marquez* | 252 | ✓ | — |
| 17 | The Curious Incident of the Dog in the Night-Time | The Curious Incident of the Dog in the Night-Time by *Mark Haddon* | 204 | ✓ | The Curious Incident of the Dog in the Night-Time : A Novel (82) |
| 18 | Crime and Punishment | Crime and Punishment by *Fyodor Dostoevsky* | 140 | ✓ | — |
| 19 | War and Peace | Peace Like a River by *Leif Enger* | 165 | **✗** | A Separate Peace (118), War and Peace (Wordsworth Classics) (100), SILENT NIGHT : The Story of the World War I Christmas Truce (53) |
| 20 | The Master and Margarita | The Master and Margarita by *Mikhail Bulgakov* | 65 | ✓ | — |

## 2 · Queried but not in the set

`resolves to` is what the app answers. `the floor is hiding` is the group M23.10.5 adds to the picker: books the query matches **better** than anything the demo can answer, which the search box has dropped silently since M13.

| typed | resolves to | readers | the floor is hiding |
|---|---|--:|---|
| The Kite Runner | The Wailing Wind by *Tony Hillerman* | 116 | The Kite Runner by *Khaled Hosseini* (39) · The Kite Rider by *Geraldine McCaughrean* (7) · The Runaway Kite by *Walt Disney* (3) |
| The Purpose Driven Life | The Purpose-Driven Life: What on Earth Am I Here For? by *Rick Warren* | 79 | The Power of Purpose: Creating Meaning in Your Life and Work by *Richard J. Leider* (4) · Living Life on Purpose by *G. Anderson* (2) · The Call : Finding and Fulfilling the Central Purpose of Your Life by *OS Guinness* (1) · The On-Purpose Person: Making Your Life Make Sense : A Modern Parable by *Kevin W. McCarthy* (1) · The Purpose Driven Life: What on Earth Am I Here For? by *Rick Warren* (1) |
| The Brothers Karamazov | Into the Wild by *Jon Krakauer* | 251 | The Brothers Karamazov by *FYODOR DOSTOEVSKY* (36) · Brothers Karamazov by *Fedor Dostoevsky* (26) · The Karamazov Brothers by *Fyodor Dostoevsky* (4) · Los Hermanos Karamazov by *Fyodor M. Dostoevsky* (1) · The Brothers Karamazov: A Novel in Four Parts With Epilogue by *Fyodor Dostoyevsky* (1) |

## 2b · The work keys M23.10.7 has to cite

| work id | readers | askable at the floor |
|---|--:|:-:|
| `the kite runner|hosseini` | 39 | no |
| `the purpose-driven life: what on earth am i here for?|warren` | 79 | yes |
| `the purpose driven life: what on earth am i here for?|warren` | 1 | no |
| `the brothers karamazov|dostoevsky` | 36 | no |
| `brothers karamazov|dostoevsky` | 26 | no |
| `the curious incident of the dog in the night-time|haddon` | 204 | yes |
| `the curious incident of the dog in the night-time: a novel|haddon` | 82 | yes |

## 3 · The two lists, anchor by anchor

### 1. The Lovely Bones: A Novel by Alice Sebold

typed *The Lovely Bones* · `the lovely bones: a novel|sebold` · **1,295 readers**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | Lucky : A Memoir by *Alice Sebold* | 134 | Lucky : A Memoir by *Alice Sebold* | 134 |
| 2 | The Nanny Diaries: A Novel by *Emma McLaughlin* | 204 | The Nanny Diaries: A Novel by *Emma McLaughlin* | 204 |
| 3 | Lost Girls by *Andrew Pyper* | 26 | The Secret Life of Bees by *Sue Monk Kidd* | 185 |
| 4 | The Berenstain Bears and Too Much Vacation by *Stan Berenstain* | 9 | She's Come Undone by *Wally Lamb* | 200 |
| 5 | The Four Agreements: A Practical Guide to Personal Freedom by *Don Miguel Ruiz* | 34 | The Da Vinci Code by *Dan Brown* | 195 |
| 6 | The Dive From Clausen's Pier by *ANN PACKER* | 38 | Divine Secrets of the Ya-Ya Sisterhood: A Novel by *Rebecca Wells* | 182 |
| 7 | He Sees You When You're Sleeping : A Novel by *Carol Higgins Clark* | 26 | Life of Pi by *Yann Martel* | 159 |
| 8 | Of Love and Shadows by *Isabel Allende* | 18 | White Oleander : A Novel by *Janet Fitch* | 164 |
| 9 | Silent Spring by *Rachel Carson* | 9 | Midwives: A Novel by *Chris Bohjalian* | 119 |
| 10 | One Thousand White Women : The Journals of May Dodd: A Novel by *Jim Fergus* | 22 | The Notebook by *Nicholas Sparks* | 149 |

### 2. The Da Vinci Code by Dan Brown

typed *The Da Vinci Code* · `the da vinci code|brown` · **905 readers**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | Angels & Demons by *Dan Brown* | 216 | Angels & Demons by *Dan Brown* | 216 |
| 2 | Stupid White Men : ...And Other Sorry Excuses for the State of the Nation! by *Michael Moore* | 6 | The Lovely Bones: A Novel by *Alice Sebold* | 195 |
| 3 | Digital Fortress : A Thriller by *Dan Brown* | 77 | Life of Pi by *Yann Martel* | 137 |
| 4 | Deception Point by *Dan Brown* | 66 | The Secret Life of Bees by *Sue Monk Kidd* | 141 |
| 5 | The Eight by *Katherine Neville* | 40 | Harry Potter and the Order of the Phoenix by *J. K. Rowling* | 93 |
| 6 | Life of Pi by *Yann Martel* | 137 | Balzac and the Little Chinese Seamstress : A Novel by *DAI SIJIE* | 77 |
| 7 | The Narrows: A Novel by *Michael Connelly* | 15 | The Five People You Meet in Heaven by *Mitch Albom* | 94 |
| 8 | The Last Juror by *John Grisham* | 34 | Digital Fortress : A Thriller by *Dan Brown* | 77 |
| 9 | Trojan Odyssey by *Clive Cussler* | 8 | The Time Traveler's Wife by *Audrey Niffenegger* | 60 |
| 10 | The Dante Club : A Novel by *MATTHEW PEARL* | 24 | The Crimson Petal and the White by *Michel Faber* | 62 |

### 3. Harry Potter and the Sorcerer's Stone by J. K. Rowling

typed *Harry Potter and the Sorcerer's Stone* · `harry potter and the sorcerer's stone|rowling` · **832 readers**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | Harry Potter and the Chamber of Secrets by *J. K. Rowling* | 343 | Harry Potter and the Chamber of Secrets by *J. K. Rowling* | 343 |
| 2 | Harry Potter and the Prisoner of Azkaban by *J. K. Rowling* | 266 | Harry Potter and the Prisoner of Azkaban by *J. K. Rowling* | 266 |
| 3 | Harry Potter and the Goblet of Fire by *J. K. Rowling* | 224 | Harry Potter and the Goblet of Fire by *J. K. Rowling* | 224 |
| 4 | Harry Potter and the Order of the Phoenix by *J. K. Rowling* | 149 | Harry Potter and the Order of the Phoenix by *J. K. Rowling* | 149 |
| 5 | Fantastic Beasts and Where to Find Them by *J. K. Rowling* | 12 | The Fellowship of the Ring by *J.R.R. TOLKIEN* | 96 |
| 6 | Quidditch Through the Ages by *J. K. Rowling* | 11 | The Return of the King by *J.R.R. TOLKIEN* | 76 |
| 7 | Don't Know Much About History: Everything You Need to Know About American History but Never Learned by *Kenneth Davis* | 12 | The Two Towers by *J.R.R. TOLKIEN* | 79 |
| 8 | Fever 1793 by *Laurie Halse Anderson* | 5 | The Golden Compass by *PHILIP PULLMAN* | 79 |
| 9 | The 10th Kingdom by *Kathryn Wesley* | 8 | The Hobbit : The Enchanting Prelude to The Lord of the Rings by *J.R.R. TOLKIEN* | 74 |
| 10 | Odd Girl Out: The Hidden Culture of Aggression in Girls by *Rachel Simmons* | 9 | Anne of the Island by *Lucy Maud Montgomery* | 45 |

### 4. Bridget Jones's Diary by Helen Fielding

typed *Bridget Jones's Diary* · `bridget jones's diary|fielding` · **772 readers**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | Bridget Jones: The Edge of Reason by *Helen Fielding* | 145 | Bridget Jones: The Edge of Reason by *Helen Fielding* | 145 |
| 2 | The Lost Boy by *Dave Pelzer* | 9 | Divine Secrets of the Ya-Ya Sisterhood: A Novel by *Rebecca Wells* | 146 |
| 3 | Sex & the City by *Candace Bushnell* | 26 | Where the Heart Is by *Billie Letts* | 122 |
| 4 | Does My Bum Look Big in This? by *Arabella Weir* | 7 | Midwives: A Novel by *Chris Bohjalian* | 94 |
| 5 | The Diary of a Nobody by *George Grossmith* | 8 | The Nanny Diaries: A Novel by *Emma McLaughlin* | 134 |
| 6 | About a Boy Uk by *Nick Hornby* | 16 | Confessions of a Shopaholic by *SOPHIE KINSELLA* | 97 |
| 7 | The Tenant of Wildfell Hall by *Anne Bronte* | 11 | The Girls' Guide to Hunting and Fishing by *Melissa Bank* | 109 |
| 8 | Cause Celeb by *Helen Fielding* | 37 | She's Come Undone by *Wally Lamb* | 140 |
| 9 | A Cup of Tea by *Amy Ephron* | 18 | Good in Bed by *Jennifer Weiner* | 100 |
| 10 | Lizard by *Banana Yoshimoto* | 12 | Message in a Bottle by *Nicholas Sparks* | 99 |

### 5. Girl with a Pearl Earring by Tracy Chevalier

typed *Girl with a Pearl Earring* · `girl with a pearl earring|chevalier` · **647 readers**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | Falling Angels by *Tracy Chevalier* | 55 | The Secret Life of Bees by *Sue Monk Kidd* | 134 |
| 2 | The Virgin Blue by *Tracy Chevalier* | 44 | The Red Tent by *Anita Diamant* | 119 |
| 3 | The Secret Life of Bees by *Sue Monk Kidd* | 134 | Life of Pi by *Yann Martel* | 109 |
| 4 | Girl in Hyacinth Blue by *Susan Vreeland* | 62 | House of Sand and Fog by *Andre Dubus III* | 102 |
| 5 | The Dante Club : A Novel by *MATTHEW PEARL* | 19 | Girl in Hyacinth Blue by *Susan Vreeland* | 62 |
| 6 | Straight Man by *Richard Russo* | 3 | Falling Angels by *Tracy Chevalier* | 55 |
| 7 | Tulip Fever by *DEBORAH MOGGACH* | 18 | Bel Canto: A Novel by *Ann Patchett* | 76 |
| 8 | The Red Tent by *Anita Diamant* | 119 | The Crimson Petal and the White by *Michel Faber* | 56 |
| 9 | The Lady and the Unicorn by *Tracy Chevalier* | 18 | The Girls' Guide to Hunting and Fishing by *Melissa Bank* | 94 |
| 10 | The Picture of Dorian Gray by *Oscar Wilde* | 37 | Divine Secrets of the Ya-Ya Sisterhood: A Novel by *Rebecca Wells* | 115 |

### 6. Interview with the Vampire by Anne Rice

typed *Interview with the Vampire* · `interview with the vampire|rice` · **521 readers**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | The Vampire Lestat by *ANNE RICE* | 169 | The Vampire Lestat by *ANNE RICE* | 169 |
| 2 | The Queen of the Damned by *Anne Rice* | 133 | The Queen of the Damned by *Anne Rice* | 133 |
| 3 | The Tale of the Body Thief by *Anne Rice* | 106 | The Tale of the Body Thief by *Anne Rice* | 106 |
| 4 | The Vampire Armand by *Anne Rice* | 25 | The Mummy or Ramses the Damned by *Anne Rice* | 60 |
| 5 | Memnoch the Devil by *Anne Rice* | 39 | Pet Sematary by *Stephen King* | 67 |
| 6 | The Mummy or Ramses the Damned by *Anne Rice* | 60 | Silence of the Lambs by *Thomas Harris* | 71 |
| 7 | Vittorio the Vampire: New Tales of the Vampires by *Anne Rice* | 45 | The Witching Hour by *ANNE RICE* | 70 |
| 8 | Pandora: New Tales of the Vampires by *Anne Rice* | 51 | Jurassic Park by *Michael Crichton* | 92 |
| 9 | Lasher: Lives of the Mayfair Witches by *Anne Rice* | 36 | Vittorio the Vampire: New Tales of the Vampires by *Anne Rice* | 45 |
| 10 | The Witching Hour by *ANNE RICE* | 70 | Memnoch the Devil by *Anne Rice* | 39 |

### 7. To Kill a Mockingbird by Harper Lee

typed *To Kill a Mockingbird* · `to kill a mockingbird|lee` · **495 readers**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | Of Mice and Men by *John Steinbeck* | 42 | The Joy Luck Club by *Amy Tan* | 91 |
| 2 | CRY THE BELOVED COUNTRY by *Alan Paton* | 15 | Wuthering Heights by *EMILY BRONTE* | 81 |
| 3 | The Crucible: A Play in Four Acts by *Arthur Miller* | 22 | Lord of the Flies by *William Gerald Golding* | 60 |
| 4 | One Flew Over the Cuckoo's Nest by *Ken Kesey* | 35 | The Catcher in the Rye by *J.D. Salinger* | 77 |
| 5 | Pygmalion : A Romance in Five Acts by *George Bernard Shaw* | 5 | The Firm by *John Grisham* | 83 |
| 6 | The Catcher in the Rye by *J.D. Salinger* | 77 | Snow Falling on Cedars by *David Guterson* | 90 |
| 7 | Call of the Wild by *Jack London* | 43 | Anne of Green Gables by *L.M. MONTGOMERY* | 59 |
| 8 | The Awakening by *Kate Chopin* | 18 | A Time to Kill by *JOHN GRISHAM* | 80 |
| 9 | The True Confessions of Charlotte Doyle by *Avi* | 24 | She's Come Undone by *Wally Lamb* | 102 |
| 10 | Catcher in the Rye by *Salinger* | 9 | The Horse Whisperer by *Nicholas Evans* | 74 |

### 8. The Hobbit : The Enchanting Prelude to The Lord of the Rings by J.R.R. TOLKIEN

typed *The Hobbit* · `the hobbit: the enchanting prelude to the lord of the rings|tolkien` · **281 readers**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | The Return of the King by *J.R.R. TOLKIEN* | 68 | The Two Towers by *J.R.R. TOLKIEN* | 75 |
| 2 | The Two Towers by *J.R.R. TOLKIEN* | 75 | The Return of the King by *J.R.R. TOLKIEN* | 68 |
| 3 | The Fellowship of the Ring by *J.R.R. TOLKIEN* | 82 | The Fellowship of the Ring by *J.R.R. TOLKIEN* | 82 |
| 4 | The Silmarillion by *J.R.R. TOLKIEN* | 19 | Harry Potter and the Sorcerer's Stone by *J. K. Rowling* | 74 |
| 5 | The Hobbit: or There and Back Again by *J.R.R. Tolkien* | 9 | Harry Potter and the Chamber of Secrets by *J. K. Rowling* | 56 |
| 6 | Chicken Soup for the Kid's Soul : 101 Stories of Courage, Hope and Laughter by *Jack Canfield* | 9 | The Gunslinger by *Stephen King* | 31 |
| 7 | The Book of Lost Tales 1 by *J. R. R. Tolkien* | 5 | Harry Potter and the Prisoner of Azkaban by *J. K. Rowling* | 43 |
| 8 | Eight Tales of Terror by *Edgar Allan Poe* | 6 | The Waste Lands by *Stephen King* | 20 |
| 9 | The Tolkien Reader by *J. R. R. Tolkien* | 4 | Harry Potter and the Goblet of Fire by *J. K. Rowling* | 39 |
| 10 | The Worst-Case Scenario Survival Handbook by *Joshua Piven* | 7 | Dune by *Frank Herbert* | 31 |

### 9. Dune by Frank Herbert

typed *Dune* · `dune|herbert` · **257 readers**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | Dune Messiah by *Frank Herbert* | 54 | Children of Dune by *Frank Herbert* | 53 |
| 2 | Children of Dune by *Frank Herbert* | 53 | Dune Messiah by *Frank Herbert* | 54 |
| 3 | God Emperor of Dune by *Frank Herbert* | 39 | God Emperor of Dune by *Frank Herbert* | 39 |
| 4 | Heretics of Dune by *Frank Herbert* | 26 | Heretics of Dune by *Frank Herbert* | 26 |
| 5 | Chapterhouse Dune by *Frank Herbert* | 27 | Chapterhouse Dune by *Frank Herbert* | 27 |
| 6 | House Atreides by *Brian Herbert* | 17 | Silence of the Lambs by *Thomas Harris* | 40 |
| 7 | The Time Machine by *H. G. Wells* | 14 | Interview with the Vampire by *Anne Rice* | 52 |
| 8 | Cradle by *Gentry Lee* | 3 | House Harkonnen by *Brian Herbert* | 16 |
| 9 | 2061: Odyssey Three by *Arthur C. Clarke* | 14 | The Terminal Man by *Michael Crichton* | 23 |
| 10 | House Harkonnen by *Brian Herbert* | 16 | Jurassic Park by *Michael Crichton* | 47 |

### 10. Fight Club by Chuck Palahniuk

typed *Fight Club* · `fight club|palahniuk` · **102 readers**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | Choke by *CHUCK PALAHNIUK* | 24 | Invisible Monsters by *Chuck Palahniuk* | 20 |
| 2 | Invisible Monsters by *Chuck Palahniuk* | 20 | Choke by *CHUCK PALAHNIUK* | 24 |
| 3 | Survivor : A Novel by *CHUCK PALAHNIUK* | 17 | Survivor : A Novel by *CHUCK PALAHNIUK* | 17 |
| 4 | Trainspotting by *Irvine Welsh* | 7 | Lullaby: A Novel by *Chuck Palahniuk* | 16 |
| 5 | The Tesseract by *Alex Garland* | 4 | Brave New World by *Aldous Huxley* | 17 |
| 6 | Lullaby: A Novel by *Chuck Palahniuk* | 16 | Trainspotting by *Irvine Welsh* | 7 |
| 7 | South of the Border, West of the Sun by *Haruki Murakami* | 2 | House of Leaves by *Mark Z. Danielewski* | 11 |
| 8 | All Families Are Psychotic by *Douglas Coupland* | 2 | Barrel Fever : Stories and Essays by *David Sedaris* | 10 |
| 9 | A Clockwork Orange by *Anthony Burgess* | 4 | A Short History of Nearly Everything by *Bill Bryson* | 7 |
| 10 | Helter Skelter: The True Story of the Manson Murders by *Vincent Bugliosi* | 1 | Big Fish by *Daniel Wallace* | 7 |

### 11. Secrets by DANIELLE STEEL

typed *Guns, Germs, and Steel* · `secrets|steel` · **98 readers**  ·  ⚠ **not the book this query names**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | Fine Things by *Danielle Steel* | 29 | Changes by *Danielle Steel* | 29 |
| 2 | Daddy by *DANIELLE STEEL* | 31 | The Ring by *Danielle Steel* | 24 |
| 3 | Wanderlust by *Danielle Steel* | 27 | The Promise by *Danielle Steel* | 23 |
| 4 | Once in a Lifetime by *Danielle Steel* | 18 | Fine Things by *Danielle Steel* | 29 |
| 5 | Summer's End by *Danielle Steel* | 20 | Daddy by *DANIELLE STEEL* | 31 |
| 6 | Crossings by *DANIELLE STEEL* | 22 | Message from Nam by *Danielle Steel* | 27 |
| 7 | Zoya by *Danielle Steel* | 28 | Wanderlust by *Danielle Steel* | 27 |
| 8 | Kaleidoscope by *Danielle Steel* | 28 | Star by *Danielle Steel* | 24 |
| 9 | Star by *Danielle Steel* | 24 | Zoya by *Danielle Steel* | 28 |
| 10 | The Ring by *Danielle Steel* | 24 | Thurston House by *Danielle Steel* | 24 |

### 11b. Guns, Germs, and Steel: The Fates of Human Societies by Jared Diamond

the anchor the set means by *Guns, Germs, and Steel*, reached by work id because the search box does not reach it · `guns, germs, and steel: the fates of human societies|diamond` · **67 readers**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | The Summer Tree by *Guy Gavriel Kay* | 4 | Krakatoa : The Day the World Exploded: August 27, 1883 by *Simon Winchester* | 7 |
| 2 | Microserfs by *Douglas Coupland* | 5 | The Coffee Trader : A Novel by *DAVID LISS* | 5 |
| 3 | Chaos: Making a New Science by *James Gleick* | 1 | The Botany of Desire: A Plant's-Eye View of the World by *Michael Pollan* | 5 |
| 4 | Amsterdam : A Novel by *IAN MCEWAN* | 6 | Last Days of Summer by *Steve Kluger* | 4 |
| 5 | Quicksilver : Volume One of The Baroque Cycle by *Neal Stephenson* | 2 | Becoming Madame Mao by *Anchee Min* | 4 |
| 6 | LIFE AFTER GOD : LIFE AFTER GOD by *Douglas Coupland* | 3 | The Gifts of the Jews : How a Tribe of Desert Nomads Changed the Way Everyone Thinks and Feels by *Thomas Cahill* | 4 |
| 7 | Krakatoa : The Day the World Exploded: August 27, 1883 by *Simon Winchester* | 7 | The Summer Tree by *Guy Gavriel Kay* | 4 |
| 8 | A Secret History : The Book Of Ash, #1 by *Mary Gentle* | 1 | Fingersmith by *Sarah Waters* | 7 |
| 9 | How to Be Alone: Essays by *Jonathan Franzen* | 1 | A Cook's Tour : Global Adventures in Extreme Cuisines by *Anthony Bourdain* | 4 |
| 10 | Native Speaker by *Chang-Rae Lee* | 3 | Middlesex: A Novel by *Jeffrey Eugenides* | 10 |

### 12. Life of Pi by Yann Martel

typed *Life of Pi* · `life of pi|martel` · **658 readers**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | The Piano Shop on the Left Bank: Discovering a Forgotten Passion in a Paris Atelier by *Thaddeus Carhart* | 3 | The Secret Life of Bees by *Sue Monk Kidd* | 135 |
| 2 | The Curious Incident of the Dog in the Night-Time by *Mark Haddon* | 53 | Balzac and the Little Chinese Seamstress : A Novel by *DAI SIJIE* | 81 |
| 3 | Everything Is Illuminated : A Novel by *Jonathan Safran Foer* | 23 | The Crimson Petal and the White by *Michel Faber* | 64 |
| 4 | Three Junes by *JULIA GLASS* | 55 | The Da Vinci Code by *Dan Brown* | 137 |
| 5 | Oryx and Crake by *Margaret Atwood* | 38 | The Lovely Bones: A Novel by *Alice Sebold* | 159 |
| 6 | The Eyre Affair: A Novel by *Jasper Fforde* | 43 | Good in Bed by *Jennifer Weiner* | 96 |
| 7 | The Corrections: A Novel by *Jonathan Franzen* | 30 | Girl with a Pearl Earring by *Tracy Chevalier* | 109 |
| 8 | The Da Vinci Code by *Dan Brown* | 137 | Middlesex: A Novel by *Jeffrey Eugenides* | 66 |
| 9 | South of the Border, West of the Sun by *Haruki Murakami* | 5 | The Nanny Diaries: A Novel by *Emma McLaughlin* | 118 |
| 10 | The Time Traveler's Wife by *Audrey Niffenegger* | 48 | The Red Tent by *Anita Diamant* | 108 |

### 13. Tuesdays with Morrie: An Old Man, a Young Man, and Life's Greatest Lesson by MITCH ALBOM

typed *Tuesdays with Morrie* · `tuesdays with morrie: an old man, a young man, and life's greatest lesson|albom` · **492 readers**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | The Five People You Meet in Heaven by *Mitch Albom* | 85 | The Five People You Meet in Heaven by *Mitch Albom* | 85 |
| 2 | Time Flies by *BILL COSBY* | 19 | Midwives: A Novel by *Chris Bohjalian* | 70 |
| 3 | The Four Agreements: A Practical Guide to Personal Freedom by *Don Miguel Ruiz* | 20 | The Secret Life of Bees by *Sue Monk Kidd* | 95 |
| 4 | Pay It Forward: A Novel by *Catherine Ryan Hyde* | 18 | The Lovely Bones: A Novel by *Alice Sebold* | 119 |
| 5 | Finding Fish by *Antwone Fisher* | 3 | She's Come Undone by *Wally Lamb* | 100 |
| 6 | Animal Liberation by *Peter Singer* | 4 | Fried Green Tomatoes at the Whistle Stop Cafe by *Fannie Flagg* | 59 |
| 7 | Cloud of Sparrows by *TAKASHI MATSUOKA* | 5 | Where the Heart Is by *Billie Letts* | 78 |
| 8 | Slander: Liberal Lies About the American Right by *ANN COULTER* | 4 | The Notebook by *Nicholas Sparks* | 79 |
| 9 | Fried Green Tomatoes at the Whistle Stop Cafe by *Fannie Flagg* | 59 | The Pilot's Wife : A Novel by *Anita Shreve* | 75 |
| 10 | Bias: A CBS Insider Exposes How the Media Distort the News by *Bernard Goldberg* | 3 | Drowning Ruth by *CHRISTINA SCHWARZ* | 58 |

### 14. Angela's Ashes (MMP) : A Memoir by Frank McCourt

typed *Angela's Ashes* · `angela's ashes (mmp): a memoir|mccourt` · **326 readers**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | Tis : A Memoir by *Frank McCourt* | 43 | She's Come Undone by *Wally Lamb* | 76 |
| 2 | The Cider House Rules by *John Irving* | 44 | Tis : A Memoir by *Frank McCourt* | 43 |
| 3 | The End Of The Dream The Golden Boy Who Never Grew Up : Ann Rules Crime Files Volume 5 by *Ann Rule* | 8 | The Kitchen God's Wife by *Amy Tan* | 44 |
| 4 | A Rose For Her Grave & Other True Cases by *Ann Rule* | 11 | The Notebook by *Nicholas Sparks* | 60 |
| 5 | Secret History by *DONNA TARTT* | 28 | The Bonesetter's Daughter by *Amy Tan* | 46 |
| 6 | The Lost Boy: A Foster Child's Search for the Love of a Family by *Dave Pelzer* | 17 | The Cider House Rules by *John Irving* | 44 |
| 7 | Under the Tuscan Sun by *Frances Mayes* | 21 | Saint Maybe by *ANNE TYLER* | 27 |
| 8 | The Bad Girl's Guide to the Open Road by *Cameron Tuttle* | 4 | Where the Red Fern Grows by *Wilson Rawls* | 25 |
| 9 | Krakatoa : The Day the World Exploded: August 27, 1883 by *Simon Winchester* | 7 | Message in a Bottle by *Nicholas Sparks* | 47 |
| 10 | The Five Love Languages: Five Love Languages by *Gary Chapman* | 4 | Snow Falling on Cedars by *David Guterson* | 54 |

### 15. Love in the Time of Cholera by Gabriel Garcia Marquez

typed *Love in the Time of Cholera* · `love in the time of cholera|marquez` · **261 readers**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | One Hundred Years of Solitude by *Gabriel Garcia Marquez* | 35 | Confessions of an Ugly Stepsister : A Novel by *Gregory Maguire* | 36 |
| 2 | Chronicle of a Death Foretold by *GABRIEL GARCIA MARQUEZ* | 8 | Balzac and the Little Chinese Seamstress : A Novel by *DAI SIJIE* | 38 |
| 3 | The Book of Laughter and Forgetting by *Milan Kundera* | 6 | The Red Tent by *Anita Diamant* | 59 |
| 4 | White Noise by *Don DeLillo* | 11 | One Hundred Years of Solitude by *Gabriel Garcia Marquez* | 35 |
| 5 | Unbearable Lightness of Being by *Milan Kundera* | 4 | STONES FROM THE RIVER by *Ursula Hegi* | 42 |
| 6 | Angels and Demons by *Dan Brown* | 1 | Snow Falling on Cedars by *David Guterson* | 55 |
| 7 | The Map That Changed the World : William Smith and the Birth of Modern Geology by *Simon Winchester* | 8 | Pigs in Heaven by *Barbara Kingsolver* | 33 |
| 8 | Prodigal Summer by *Barbara Kingsolver* | 18 | White Oleander : A Novel by *Janet Fitch* | 55 |
| 9 | Something Wicked This Way Comes by *Ray Bradbury* | 6 | The Bean Trees by *Barbara Kingsolver* | 40 |
| 10 | The Unbearable Lightness of Being by *Milan Kundera* | 8 | Wicked: The Life and Times of the Wicked Witch of the West by *Gregory Maguire* | 39 |

### 16. One Hundred Years of Solitude by Gabriel Garcia Marquez

typed *One Hundred Years of Solitude* · `one hundred years of solitude|marquez` · **252 readers**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | Love in the Time of Cholera by *Gabriel Garcia Marquez* | 35 | Balzac and the Little Chinese Seamstress : A Novel by *DAI SIJIE* | 37 |
| 2 | White Noise by *Don DeLillo* | 9 | Love in the Time of Cholera by *Gabriel Garcia Marquez* | 35 |
| 3 | Of Love and Other Demons by *Gabriel Garcia Marquez* | 8 | Girl with a Pearl Earring by *Tracy Chevalier* | 53 |
| 4 | The God of Small Things by *Arundhati Roy* | 31 | Life of Pi by *Yann Martel* | 53 |
| 5 | Midnight's Children by *Salman Rushdie* | 9 | Brave New World by *Aldous Huxley* | 30 |
| 6 | The Jungle Books by *Rudyard Kipling* | 2 | Fortune's Rocks: A Novel by *Anita Shreve* | 32 |
| 7 | Feel the Fear and Do It Anyway by *Susan Jeffers* | 1 | The Da Vinci Code by *Dan Brown* | 58 |
| 8 | Brothers Karamazov by *Fedor Dostoevsky* | 3 | Girl in Hyacinth Blue by *Susan Vreeland* | 29 |
| 9 | One Day in the Life of Ivan Denisovich by *Alexander Solzhenitsyn* | 8 | Vinegar Hill by *A. Manette Ansay* | 33 |
| 10 | Mary, Called Magdalene by *Margaret George* | 9 | Angels & Demons by *Dan Brown* | 48 |

### 17. The Curious Incident of the Dog in the Night-Time by Mark Haddon

typed *The Curious Incident of the Dog in the Night-Time* · `the curious incident of the dog in the night-time|haddon` · **204 readers**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | The Little Friend by *Donna Tartt* | 21 | The Time Traveler's Wife by *Audrey Niffenegger* | 30 |
| 2 | Brick Lane: A Novel by *Monica Ali* | 16 | Life of Pi by *Yann Martel* | 53 |
| 3 | Vernon God Little by *DBC Pierre* | 8 | The Dogs of Babel by *Carolyn Parkhurst* | 23 |
| 4 | Life of Pi by *Yann Martel* | 53 | Brick Lane: A Novel by *Monica Ali* | 16 |
| 5 | The Piano Tuner : A Novel by *DANIEL MASON* | 8 | Oryx and Crake by *Margaret Atwood* | 23 |
| 6 | Dress Your Family in Corduroy and Denim by *David Sedaris* | 9 | Middlesex: A Novel by *Jeffrey Eugenides* | 30 |
| 7 | Three Junes by *JULIA GLASS* | 26 | The No. 1 Ladies' Detective Agency by *Alexander McCall Smith* | 43 |
| 8 | The Autobiography of Miss Jane Pittman by *ERNEST J. GAINES* | 4 | Morality for Beautiful Girls by *Alexander McCall Smith* | 21 |
| 9 | The Devil in the White City : Murder, Magic, and Madness at the Fair That Changed America by *ERIK LARSON* | 14 | Three Junes by *JULIA GLASS* | 26 |
| 10 | Amsterdam : A Novel by *IAN MCEWAN* | 7 | The Little Friend by *Donna Tartt* | 21 |

### 18. Crime and Punishment by Fyodor Dostoevsky

typed *Crime and Punishment* · `crime and punishment|dostoevsky` · **140 readers**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | The Last Days of Socrates by *Plato* | 4 | Bleak House by *Charles Dickens* | 13 |
| 2 | Madame Bovary by *Gustave Flaubert* | 20 | Brave New World by *Aldous Huxley* | 24 |
| 3 | Heart of Darkness by *Joseph Conrad* | 15 | Emma by *Jane Austen* | 26 |
| 4 | War and Peace by *Leo Tolstoy* | 14 | Wuthering Heights by *EMILY BRONTE* | 33 |
| 5 | One Day in the Life of Ivan Denisovich by *Alexander Solzhenitsyn* | 10 | Madame Bovary by *Gustave Flaubert* | 20 |
| 6 | The Trial by *Franz Kafka* | 6 | Anna Karenina by *Leo Tolstoy* | 21 |
| 7 | Fathers and Sons by *Ivan Sergeevich Turgenev* | 4 | Heart of Darkness by *Joseph Conrad* | 15 |
| 8 | Don Quixote by *Miguel De Cervantes Saavedra* | 2 | Candide by *Francois M. Voltaire* | 17 |
| 9 | Doctor Zhivago by *BORIS PASTERNAK* | 10 | Hard Times by *CHARLES DICKENS* | 13 |
| 10 | White Noise by *Don DeLillo* | 8 | The Plague by *ALBERT CAMUS* | 11 |

### 19. Peace Like a River by Leif Enger

typed *War and Peace* · `peace like a river|enger` · **165 readers**  ·  ⚠ **not the book this query names**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | Empire Falls by *Richard Russo* | 38 | Empire Falls by *Richard Russo* | 38 |
| 2 | Dreams Of My Russian Summers: A Novel by *Andrei Makine* | 7 | Miss Julia Speaks Her Mind : A Novel by *Ann B. Ross* | 23 |
| 3 | The Virgin Blue by *Tracy Chevalier* | 18 | The Secret Life of Bees by *Sue Monk Kidd* | 49 |
| 4 | Thousand Pieces of Gold: A Biographical Novel by *Ruthanne Lum McCunn* | 9 | Plainsong by *KENT HARUF* | 25 |
| 5 | The Dive From Clausen's Pier by *ANN PACKER* | 16 | Crow Lake by *Mary Lawson* | 15 |
| 6 | The Story of Lucy Gault by *William Trevor* | 4 | Bel Canto: A Novel by *Ann Patchett* | 32 |
| 7 | The World Below by *SUE MILLER* | 10 | Three Junes by *JULIA GLASS* | 25 |
| 8 | Atonement : A Novel by *IAN MCEWAN* | 27 | Felicia's Journey by *William Trevor* | 14 |
| 9 | Blessings : A Novel by *ANNA QUINDLEN* | 5 | I Capture the Castle by *Dodie Smith* | 20 |
| 10 | Slammerkin by *Emma Donoghue* | 15 | Thousand Pieces of Gold: A Biographical Novel by *Ruthanne Lum McCunn* | 9 |

### 19b. War and Peace by Leo Tolstoy

the anchor the set means by *War and Peace*, reached by work id because the search box does not reach it · `war and peace|tolstoy` · **100 readers**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | The Brothers Karamazov by *FYODOR DOSTOEVSKY* | 6 | Anna Karenina by *Leo Tolstoy* | 17 |
| 2 | Crime and Punishment by *Fyodor Dostoevsky* | 14 | The Woman in White by *Wilkie Collins* | 12 |
| 3 | Anna Karenina by *Leo Tolstoy* | 17 | Count of Monte Cristo by *Alexandre Dumas* | 6 |
| 4 | One Day in the Life of Ivan Denisovich by *Alexander Solzhenitsyn* | 5 | Crime and Punishment by *Fyodor Dostoevsky* | 14 |
| 5 | The Woman in White by *Wilkie Collins* | 12 | The Satanic Verses by *Salman Rushdie* | 9 |
| 6 | Count of Monte Cristo by *Alexandre Dumas* | 6 | Tess of the D'Urbervilles by *Thomas Hardy* | 12 |
| 7 | Shirley by *Charlotte Bronte* | 4 | The Man in the Iron Mask by *Alexandre Dumas* | 8 |
| 8 | Twenty Thousand Leagues Under the Sea by *Jules Verne* | 4 | Villette by *Charlotte Bronte* | 7 |
| 9 | Kidnapped by *Robert Louis Stevenson* | 6 | The Death of Ivan Ilyich by *LEO TOLSTOY* | 5 |
| 10 | The Count of Monte Cristo by *Alexandre Dumas* | 6 | Ulysses by *James Joyce* | 8 |

### 20. The Master and Margarita by Mikhail Bulgakov

typed *The Master and Margarita* · `the master and margarita|bulgakov` · **65 readers**

| # | A · Matrix factorization (ALS) | co | B · Item-based collaborative filtering | co |
|---|---|---:|---|---:|
| 1 | GARDEN OF EDEN by *Ernest Hemingway* | 6 | GARDEN OF EDEN by *Ernest Hemingway* | 6 |
| 2 | Baltasar and Blimunda by *Jose Saramago* | 3 | Son of the Shadows by *Juliet Marillier* | 5 |
| 3 | The Fencing Master: A Novel by *Arturo Pérez-Reverte* | 3 | If on a Winter's Night a Traveler by *Italo Calvino* | 6 |
| 4 | Tan Veloz Como El Deseo by *Laura Esquivel* | 4 | The Book of Laughter and Forgetting by *Milan Kundera* | 6 |
| 5 | The Infinite Plan : A Novel by *Isabel Allende* | 4 | The New York Trilogy: City of Glass, Ghosts, the Locked Room by *Paul Auster* | 6 |
| 6 | The Unbearable Lightness of Being : A Novel by *Milan Kundera* | 3 | Crime and Punishment by *Fyodor Dostoevsky* | 9 |
| 7 | Holy Fools : A Novel by *Joanne Harris* | 3 | Lolita by *VLADIMIR NABOKOV* | 9 |
| 8 | Dona Flor and Her Two Husbands by *Jorge Amado* | 4 | The Grapes of Wrath: John Steinbeck Centennial Edition by *John Steinbeck* | 6 |
| 9 | The Virgin's Knot by *Holly Payne* | 2 | Balzac and the Little Chinese Seamstress : A Novel by *DAI SIJIE* | 12 |
| 10 | The New York Trilogy: City of Glass, Ghosts, the Locked Room by *Paul Auster* | 6 | Dona Flor and Her Two Husbands by *Jorge Amado* | 4 |

## 4 · Work-key splits

A *split* here means two nameable work ids that differ only by a subtitle, a parenthetical, a leading article or internal punctuation — two rows the catalogue treats as separate books when a reader would call them one. The classes are open items in this ledger and had no named instance until this set was read.

### The anchors' own siblings

Seeded from the twenty works the set names plus the keys M23.10.7 cites, so a split shows up here whether or not the search box reaches either half.

| seed | readers | the other half | readers | together | class |
|---|--:|---|--:|--:|---|
| `the lovely bones: a novel|sebold` | 1,295 | `the lovely bones|sebold` | 103 | 1,398 | subtitle |
| `harry potter and the sorcerer's stone|rowling` | 832 | `harry potter and the sorcerer's stone: a deluxe pop-up book|rowling` | 2 | 834 | subtitle |
| `bridget jones's diary|fielding` | 772 | `bridget jones's diary: a novel|fielding` | 3 | 775 | subtitle |
| `interview with the vampire|rice` | 521 | `interview with the vampire: anniversary edition|rice` | 15 | 536 | subtitle |
| `to kill a mockingbird|lee` | 495 | `to kill a mockingbird: the 40th anniversary edition of the pulitzer prize-winning novel|lee` | 11 | 506 | subtitle |
| `the hobbit: the enchanting prelude to the lord of the rings|tolkien` | 281 | `the hobbit|tolkien` | 123 | 404 | subtitle |
| `the hobbit: the enchanting prelude to the lord of the rings|tolkien` | 281 | `the hobbit: or there and back again|tolkien` | 112 | 393 | other |
| `the hobbit: the enchanting prelude to the lord of the rings|tolkien` | 281 | `hobbit|tolkien` | 1 | 282 | subtitle |
| `the hobbit: the enchanting prelude to the lord of the rings|tolkien` | 281 | `the hobbit: or, there and back again|tolkien` | 1 | 282 | other |
| `fight club|palahniuk` | 102 | `fight club: a novel|palahniuk` | 2 | 104 | subtitle |
| `guns, germs, and steel: the fates of human societies|diamond` | 67 | `guns, germs and steel: the fates of human societies|diamond` | 4 | 71 | internal punctuation |
| `life of pi|martel` | 658 | `life of pi: student edition|martel` | 15 | 673 | subtitle |
| `tuesdays with morrie: an old man, a young man, and life's greatest lesson|albom` | 492 | `tuesdays with morrie: an old man, a young man, and lifes greatest lesson|albom` | 8 | 500 | other |
| `angela's ashes (mmp): a memoir|mccourt` | 326 | `angela's ashes|mccourt` | 283 | 609 | subtitle |
| `angela's ashes (mmp): a memoir|mccourt` | 326 | `angela's ashes: a memoir|mccourt` | 224 | 550 | parenthetical |
| `the curious incident of the dog in the night-time|haddon` | 204 | `the curious incident of the dog in the night-time: a novel|haddon` | 82 | 286 | subtitle |
| `crime and punishment|dostoevsky` | 140 | `crime and punishment: the coulson translation backgrounds and sources: essays in criticism|dostoevsky` | 2 | 142 | subtitle |
| `war and peace|tolstoy` | 100 | `war and peace: the maude translation, backgrounds and sources, criticism|tolstoy` | 2 | 102 | subtitle |
| `the master and margarita|bulgakov` | 65 | `master and margarita|bulgakov` | 4 | 69 | leading article |
| `the purpose-driven life: what on earth am i here for?|warren` | 79 | `the purpose driven life: what on earth am i here for?|warren` | 1 | 80 | internal punctuation |
| `the purpose driven life: what on earth am i here for?|warren` | 1 | `the purpose-driven life: what on earth am i here for?|warren` | 79 | 80 | internal punctuation |
| `the brothers karamazov|dostoevsky` | 36 | `brothers karamazov|dostoevsky` | 26 | 62 | leading article |
| `the brothers karamazov|dostoevsky` | 36 | `the brothers karamazov: the constance garnett translation revised by ralph e. matlaw: backgrounds and sources, essays in criticism|dostoevsky` | 5 | 41 | subtitle |
| `brothers karamazov|dostoevsky` | 26 | `the brothers karamazov|dostoevsky` | 36 | 62 | leading article |
| `brothers karamazov|dostoevsky` | 26 | `the brothers karamazov: the constance garnett translation revised by ralph e. matlaw: backgrounds and sources, essays in criticism|dostoevsky` | 5 | 31 | subtitle |
| `the curious incident of the dog in the night-time: a novel|haddon` | 82 | `the curious incident of the dog in the night-time|haddon` | 204 | 286 | subtitle |

### How much each class costs, over the whole nameable catalogue

| class | split groups | works involved | groups the split silences at floor 50 |
|---|--:|--:|--:|
| subtitle | 5,543 | 11,626 | 103 |
| leading article | 2,210 | 4,694 | 60 |
| other | 1,853 | 4,379 | 18 |
| internal punctuation | 1,043 | 2,202 | 12 |
| parenthetical | 88 | 190 | 1 |
| **all classes** | **10,737** | **23,091** | **194** |

*Silenced* means every half sits below the anchor floor while their sum clears it: the book exists in the data often enough to be answerable and the key hides it. That is the same defect L89 found on *Songlines*, counted rather than instanced.

**194 silenced groups against 2,508 askable works**: merging the halves would take the askable catalogue to 2,702, **+7.7%**, without touching the floor or the engine. Stated as an upper bound on this defect and not as a proposal — the merge is a data-layer change and L67 measured what a re-key does to a neighbourhood.

## 5 · The count

Filled in by hand after reading section 3, under **L79's criterion, inherited unchanged** so this count and L89's are comparable: a slot is bad when it is the anchor itself under another edition, title or translation; a companion or *about* book rather than a comparable read; or a duplicate of another slot in the same list. Empty until someone has read it — a count written by the script that produced the lists is the thing this audit exists to avoid.

| configuration | bad slots | of | date |
|---|--:|--:|---|
| A · Matrix factorization (ALS) | **7** (4 unambiguous) | 220 | 2026-08-10 |
| B · Item-based collaborative filtering | **0** | 220 | 2026-08-10 |

**Every counted slot is named, so the count can be checked rather than believed.** All seven
are A's; B produced none of the three classes in 220 slots.

| # | anchor | slot | class | note |
|--:|---|---|---|---|
| 1 | The Hobbit | A rank 5 · *The Hobbit: or There and Back Again* (9) | **the anchor under another title** | The same book. `the hobbit: or there and back again\|tolkien`, 112 readers, a fourth half of a work the key splits four ways |
| 2 | To Kill a Mockingbird | A rank 10 · *Catcher in the Rye* — Salinger (9) | **duplicate** | Of A's own rank 6, *The Catcher in the Rye* — J.D. Salinger (77). Two work keys, one leading article and one author spelling apart |
| 3 | Love in the Time of Cholera | A rank 10 · *The Unbearable Lightness of Being* (8) | **duplicate** | Of A's own rank 5, *Unbearable Lightness of Being* (4). The leading-article class again |
| 4 | War and Peace (19b) | A rank 10 · *The Count of Monte Cristo* (6) | **duplicate** | Of A's own rank 6, *Count of Monte Cristo* (6). Same |
| 5 | Harry Potter | A rank 5 · *Fantastic Beasts and Where to Find Them* (12) | *companion* | A Hogwarts Library volume, not a novel. **Arguable**: L34's docstring treats the sibling case as a result worth keeping, so this is counted and flagged rather than counted silently |
| 6 | Harry Potter | A rank 6 · *Quidditch Through the Ages* (11) | *companion* | Same, and it is the book L34 names by title |
| 7 | The Hobbit | A rank 9 · *The Tolkien Reader* (4) | *companion* | An anthology about the anchor's author rather than a comparable read |

**Three of the four unambiguous ones are the same defect, and it is the one section 4
counts.** A duplicate slot here is never two editions with two ISBNs — the work key already
merges those. It is two *work keys*, separated by a leading article, both surviving into one
top ten. So the leading-article class does not only silence books below the floor (60 groups,
section 4); above the floor it spends a slot showing the same book twice.

**And B cannot make that mistake, for a structural reason worth saying out loud.** Two halves
of a split work share almost no readers — a reader buys one edition, not both — so their
co-occurrence is near zero and B ranks them nowhere. A's factors, fitted on the same
interactions, put them *close together*, because their reader profiles look alike even when
the readers are different people. The engine that reasons about similarity in a latent space
inherits the catalogue's duplicates; the engine that counts shared readers is blind to them.
That is the same mechanism, pointed the other way, as B's weakness in section 7.


## 6 · The evidence behind the slots, counted by the script

Not a judgment and therefore not the human's to make: these are L87's definitions applied to this anchor set, over every list in section 3.

| configuration | slots | median co-readers | thin (<5 co-readers) | zero co-readers | weakest slot |
|---|--:|--:|--:|--:|--:|
| A · Matrix factorization (ALS) | 220 | 15 | 42 (19.1%) | 0 | 1 |
| B · Item-based collaborative filtering | 220 | 46 | 6 (2.7%) | 0 | 4 |

## 7 · The verdict (hand-written, M23.10.3's gate)

*Sections 5 and 7 are filled in by hand and a re-run of the script replaces them — the same
convention as `docs/floor_band_audit.md`, whose count L89 cites.*

**The gate passes. No anchor is visibly worse under B, and B ships behind the picker with A
still the default.**

**What B is better at, and it is not close.** On ten of the twenty-two lists A puts a slot on
screen that a reader would query and B does not: *The Berenstain Bears and Too Much Vacation*
under **The Lovely Bones**, *Stupid White Men* under **The Da Vinci Code** (6 shared readers of
905), *The Lost Boy* under **Bridget Jones's Diary**, four noise slots and two companion
volumes in the tail of **Harry Potter**, the anchor's own other title plus *Chicken Soup for the
Kid's Soul* and *The Worst-Case Scenario Survival Handbook* under **The Hobbit**, *The Piano
Shop on the Left Bank* at **rank 1** of **Life of Pi** on 3 shared readers of 658, two Ann Rule
true-crime titles under **Angela's Ashes**, and — the one that would be worst in front of a user —
*Slander: Liberal Lies About the American Right* and *Bias: A CBS Insider Exposes How the Media
Distort the News* under **Tuesdays with Morrie**. Section 6 is the same finding as a number:
A's median slot rests on 15 shared readers and B's on 46, and A's thin-slot rate is **19.1%
against B's 2.7%** — an independent replication of L87's 18.2% against 1.7% on a different
anchor set and a different sampling rule.

**What A is better at, stated because the switch has to be honest in both directions.** A is
more *specific* where B is more *evidenced*. On an author or a series A stays with the author:
nine Anne Rice novels under **Interview with the Vampire** against B's six, three Tracy
Chevalier novels plus *Tulip Fever* and *Girl in Hyacinth Blue* under **Girl with a Pearl
Earring**, a science-fiction tail under **Dune**, and a Russian-literature tail under **Crime
and Punishment** (*War and Peace*, *Doctor Zhivago*, *Fathers and Sons*, *One Day in the Life
of Ivan Denisovich*) where B answers with the English canon. The two weakest slots B produces
anywhere in 220 are *Jurassic Park* at rank 8 of **Interview with the Vampire** (92 shared
readers) and *The Da Vinci Code* and *Angels & Demons* at ranks 7 and 10 of **One Hundred Years
of Solitude** (58 and 48). They are **generic, not wrong**: those readers really did read those
books, the count is printed beside the row, and a visitor can see exactly what the claim rests
on. A's weakest slots are wrong: a book three of 658 readers share, at rank 1.

**So the one sentence that sums it up:** the switch trades *how specific* the list looks for *how
much it rests on*, and the app prints the evidence beside every row, so the trade is visible
rather than asserted.

**Two things the gate found that are not about the engine at all.**

1. **Two of the twenty titles do not resolve to the book they name** (section 1). "Guns, Germs,
   and Steel" answers with *Secrets* by Danielle Steel and "War and Peace" with *Peace Like a
   River* by Leif Enger. Both are the tie rule (`LOOKUP_TIE_MARGIN`, L62) doing exactly what it
   was built to do: the right book **is** the top text match, and it is then demoted by
   readership because four Danielle Steel novels sit within 0.06 of it. This is L38's
   under-determined-query finding arriving on a famous title, it is engine-independent and
   floor-independent, and **nothing in M23.10 changes it** — the correct book is offered in the
   picker at rank 4 and rank 3 respectively. Recorded here, not fixed here: the tie rule is a
   published serving rule and a rehearsed demo runs on it.
2. **The work key splits more books than the milestone assumed** (section 4). *Angela's Ashes*
   is split **three** ways (326 + 283 + 224), not two; *The Hobbit* four; *The Lovely Bones* —
   the demo's first anchor — two, at 1,295 + 103. Across the nameable catalogue the four
   classes cover **10,737 groups and 23,091 works**, and **194 of those groups are silenced at
   the floor**: every half below 50, the sum above it. That is **+7.7%** on the askable
   catalogue, available without touching the floor or the engine.
